import argparse
import json
import os
import sys
import time

import pandas as pd
import psutil
import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers.bm25 import BM25Retriever
# from langchain_classic.retrievers import EnsembleRetriever

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
sys.path.append(project_root_dir)

from src.evaluation.retrieval import (
    evaluate_retrieval_dataframe,
    summarize_retrieval,
)
from src.evaluation.ground_truth import make_ground_truth_dataframe
from src.preprocessing.cleaner import RFPTextCleaner
from src.preprocessing.loader import extract_pdf


# =============================================================================
# 유틸 함수
# =============================================================================

def resolve_project_path(path_value: str) -> str:
    path_value = os.path.expanduser(path_value)
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(project_root_dir, path_value)


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str) -> dict:
    config_path = resolve_project_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        experiment_config = yaml.safe_load(f)

    base_config_path = experiment_config.get("base_config")
    if not base_config_path:
        return experiment_config

    with open(resolve_project_path(base_config_path), "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    return deep_merge(base_config, experiment_config)


def get_document_configs(config: dict) -> list[dict]:
    documents = config.get("documents")
    if documents:
        return documents

    return [
        {
            "document_id": "korea_portal",
            "target_document": config["project"]["target_document"],
            "raw_pdf_file": config["path"]["raw_pdf_file"],
            "pdf_pages": config["path"].get("pdf_pages"),
        }
    ]


def resolve_document_pdf_path(config: dict, document_config: dict) -> str:
    raw_pdf_file = document_config.get("raw_pdf_file")
    if raw_pdf_file:
        return resolve_project_path(raw_pdf_file)

    file_name = document_config.get("file_name")
    if not file_name:
        raise KeyError("Each document must define either raw_pdf_file or file_name.")

    file_dir = resolve_project_path(config["path"]["file_dir"])
    return os.path.join(file_dir, file_name)


def run_chunking(cleaner: RFPTextCleaner, splitter: str, md_text: str, project_name: str) -> list[str]:
    if splitter == "markdown":
        return cleaner.run_markdown_chunking(md_text, project_name=project_name)
    raise ValueError("Baseline only supports markdown chunking.")


def get_dir_size_mb(path: str) -> float:
    """디렉토리(또는 파일) 전체 크기를 MB 단위로 반환."""
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 ** 2)
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            fp = os.path.join(dirpath, fname)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 2)


# =============================================================================
# Hybrid Retriever 빌드
# Dense(Chroma) + Sparse(BM25) → EnsembleRetriever
#
# document_id 필터링 전략:
#   BM25Retriever는 필터를 지원하지 않으므로, document_id별로 독립된
#   (chroma_retriever, bm25_retriever, ensemble_retriever) 세트를 미리 만들어
#   딕셔너리에 저장해두고 쿼리 시 꺼내 쓴다.
#   dense_weight + sparse_weight = 1.0 (기본값: 0.5 / 0.5)
# =============================================================================

def build_hybrid_retrievers(
    vector_db: Chroma,
    docs_by_document_id: dict[str, list[Document]],
    retrieval_k: int,
    dense_weight: float = 0.5,
) -> dict[str, EnsembleRetriever]:
    """
    document_id별로 EnsembleRetriever를 생성해 반환한다.
    """
    sparse_weight = 1.0 - dense_weight
    retrievers: dict[str, EnsembleRetriever] = {}

    for document_id, docs in docs_by_document_id.items():
        # Dense retriever: Chroma에 document_id 필터 적용
        chroma_retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": retrieval_k,
                "filter": {"document_id": document_id},
            },
        )

        # Sparse retriever: 해당 document_id 청크만으로 BM25 인덱스 구성
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = retrieval_k

        # Ensemble: RRF(Reciprocal Rank Fusion) 기반 결합
        ensemble = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25_retriever],
            weights=[dense_weight, sparse_weight],
        )
        retrievers[document_id] = ensemble

    return retrievers


# =============================================================================
# 메인
# =============================================================================

parser = argparse.ArgumentParser(description="LangChain Hybrid embedding experiment")
parser.add_argument("--config", default="configs/experiments/bge-m3_qwen3-8B.yaml")
args = parser.parse_args()

config = load_config(args.config)

# Hybrid 가중치: YAML retrieval.dense_weight 키로 조정 가능 (기본 0.5)
dense_weight = float(config["retrieval"].get("dense_weight", 0.5))

print(
    "config loaded "
    f"(framework: langchain_hybrid, "
    f"dense_weight: {dense_weight}, sparse_weight: {1.0 - dense_weight:.1f}, "
    f"splitter: {config['preprocessing']['splitter']}, "
    f"chunk_size: {config['preprocessing']['chunk_size']}, "
    f"overlap: {config['preprocessing']['chunk_overlap']})"
)

# =============================================================================
# 1. PDF 로드 & 청킹
# =============================================================================

cleaner = RFPTextCleaner(config=config)
splitter = config["preprocessing"]["splitter"]
document_configs = get_document_configs(config)
documents = []
chunk_records = []

# document_id별 청크를 따로 보관 (BM25 인덱스 구성용)
docs_by_document_id: dict[str, list[Document]] = {}

for document_config in document_configs:
    document_id = document_config["document_id"]
    project_name = document_config["target_document"]
    pdf_path = resolve_document_pdf_path(config, document_config)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    md_text = extract_pdf(
        pdf_path,
        pages=document_config.get("pdf_pages", config["path"].get("pdf_pages")),
        image_path=resolve_project_path(config["path"].get("image_dir", "outputs/images")),
        write_images=config["path"].get("write_images", False),
    )

    chunks = run_chunking(cleaner, splitter, md_text, project_name)
    print(f"Total chunks ({document_id}): {len(chunks)}")

    doc_list = []
    for local_chunk_id, chunk in enumerate(chunks):
        global_chunk_id = len(documents)
        metadata = {
            "document_id": document_id,
            "source": project_name,
            "source_file": pdf_path,
            "chunk_id": global_chunk_id,
            "local_chunk_id": local_chunk_id,
        }
        doc = Document(page_content=chunk, metadata=metadata)
        documents.append(doc)
        doc_list.append(doc)
        chunk_records.append(
            {
                **metadata,
                "chunking_strategy": splitter,
                "chunk_length": len(chunk),
                "chunk_text": chunk,
            }
        )

    docs_by_document_id[document_id] = doc_list

print(f"Total chunks: {len(documents)}")

chunk_output_path = resolve_project_path(config["output"]["chunk_results"])
os.makedirs(os.path.dirname(chunk_output_path), exist_ok=True)
chunk_df = pd.DataFrame(chunk_records)
chunk_df.to_csv(chunk_output_path, index=False, encoding="utf-8-sig")
print(f"Chunking result saved: {chunk_output_path}")

# =============================================================================
# 2. 임베딩 모델 로드
# =============================================================================

load_dotenv()
embedding_provider = config["embedding"]["provider"]

if embedding_provider == "openai":
    from langchain_openai import OpenAIEmbeddings

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")
    embeddings_b = OpenAIEmbeddings(model=config["embedding"]["model"])

elif embedding_provider == "huggingface":
    embeddings_b = HuggingFaceEmbeddings(
        model_name=config["embedding"]["model"],
        model_kwargs={
            "device": "cuda",
            "trust_remote_code": True,
        },
        encode_kwargs={"normalize_embeddings": True},
    )
else:
    raise ValueError(f"Unknown embedding provider: {embedding_provider}")

# =============================================================================
# 3. Chroma 벡터 DB 빌드 (Dense 인덱스)
#    BM25는 인메모리라 별도 저장 불필요 — generation 단계에서 재구성
# =============================================================================

persist_db_b = resolve_project_path(config["retrieval"]["persist_directory"])
os.makedirs(persist_db_b, exist_ok=True)

process = psutil.Process(os.getpid())
ram_before_mb = process.memory_info().rss / (1024 ** 2)

start_db = time.time()

vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_b,
    persist_directory=persist_db_b,
)

embedding_build_seconds = time.time() - start_db
ram_after_mb = process.memory_info().rss / (1024 ** 2)
peak_ram_delta_mb = max(ram_after_mb - ram_before_mb, 0.0)

index_size_mb = get_dir_size_mb(persist_db_b)
embedding_build_seconds_per_chunk = (
    embedding_build_seconds / len(documents) if documents else 0.0
)

print(f"Vector DB (Chroma/Dense) built and saved ({embedding_build_seconds:.2f}s): {persist_db_b}")
print(f"  Index size on disk : {index_size_mb:.2f} MB")
print(f"  RAM delta (build)  : {peak_ram_delta_mb:.2f} MB")

# =============================================================================
# 4. Hybrid Retriever 빌드 (Dense + BM25)
# =============================================================================

retrieval_k = int(os.environ.get("RAG_RETRIEVAL_K", str(config["retrieval"].get("top_k", 4))))

hybrid_retrievers = build_hybrid_retrievers(
    vector_db=vector_db,
    docs_by_document_id=docs_by_document_id,
    retrieval_k=retrieval_k,
    dense_weight=dense_weight,
)
print(f"Hybrid retrievers built for: {list(hybrid_retrievers.keys())}")

# =============================================================================
# 5. 샘플 쿼리 테스트
# =============================================================================

query = config["retrieval"]["sample_query"]
print(f"\n[Sample query test] Question: '{query}'")
sample_document_id = config["retrieval"].get("sample_document_id", document_configs[0]["document_id"])

sample_retriever = hybrid_retrievers[sample_document_id]
retrieved_docs_sample = sample_retriever.invoke(query)[:2]

print("=" * 50)
for idx, doc in enumerate(retrieved_docs_sample):
    print(f"[Retrieved chunk {idx + 1}] (Chunk ID: {doc.metadata['chunk_id']})")
    print(doc.page_content)
    print("-" * 50)

# =============================================================================
# 6. Ground Truth 기반 Retrieval 평가
# =============================================================================

strategy_name = config["output"]["strategy_name"]
ground_truth_df = make_ground_truth_dataframe(
    [document_config["document_id"] for document_config in document_configs]
)
retrieval_rows = []

for _, row in ground_truth_df.iterrows():
    question = row["question"]
    document_id = row["document_id"]

    retriever = hybrid_retrievers[document_id]

    start_q = time.time()
    retrieved_docs = retriever.invoke(question)[:retrieval_k]
    query_latency_ms = (time.time() - start_q) * 1000

    retrieved_ranked_chunks = [
        {
            "rank": rank,
            "document_id": doc.metadata.get("document_id", ""),
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "local_chunk_id": doc.metadata.get("local_chunk_id", ""),
            "source": doc.metadata.get("source", ""),
            "chunk_text": doc.page_content,
        }
        for rank, doc in enumerate(retrieved_docs, start=1)
    ]
    retrieval_rows.append(
        {
            "strategy": strategy_name,
            "document_id": row["document_id"],
            "document_name": row["document_name"],
            "question_id": row["question_id"],
            "question": question,
            "ground_truth": row["ground_truth"],
            "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
            "retrieved_chunk_ids": ", ".join(
                str(doc.metadata.get("chunk_id", "")) for doc in retrieved_docs
            ),
            "retrieved_ranked_chunks": json.dumps(retrieved_ranked_chunks, ensure_ascii=False),
            # --- 성능 지표 ---
            "embedding_build_seconds": embedding_build_seconds,
            "embedding_build_seconds_per_chunk": embedding_build_seconds_per_chunk,
            "query_latency_ms": query_latency_ms,
            "index_size_mb": index_size_mb,
            "peak_ram_delta_mb": peak_ram_delta_mb,
            "dense_weight": dense_weight,
            "sparse_weight": 1.0 - dense_weight,
        }
    )

retrieval_df = pd.DataFrame(retrieval_rows)
evaluated_df = evaluate_retrieval_dataframe(retrieval_df)
summary_df = summarize_retrieval(evaluated_df)

# =============================================================================
# 7. 결과 저장
# =============================================================================

retrieval_output_path = resolve_project_path(config["output"]["retrieval_eval_results"])
summary_output_path = resolve_project_path(config["output"]["retrieval_eval_summary"])
os.makedirs(os.path.dirname(retrieval_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

evaluated_df.to_csv(retrieval_output_path, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_output_path, index=False, encoding="utf-8-sig")

print("\n[Retrieval evaluation complete]")
print(f"Evaluation saved : {retrieval_output_path}")
print(f"Summary saved    : {summary_output_path}")
print("\n[Retrieval evaluation summary]")
print(summary_df.to_string(index=False))
