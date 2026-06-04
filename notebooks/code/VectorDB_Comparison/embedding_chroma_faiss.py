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
from langchain_community.vectorstores import FAISS

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
# FAISS: document_id 기반 후처리 필터
# (FAISS는 메타데이터 필터를 네이티브 지원하지 않으므로 Python 레벨에서 처리)
# =============================================================================

def faiss_similarity_search_with_filter(
    vector_db: FAISS,
    query: str,
    k: int,
    filter_dict: dict,
    fetch_k_multiplier: int = 10,
) -> list:
    """
    FAISS에서 document_id 필터를 에뮬레이션한다.
    원하는 k개가 채워질 때까지 fetch_k_multiplier 배수로 더 넓게 검색 후 후처리 필터링.
    """
    fetch_k = k * fetch_k_multiplier
    candidates = vector_db.similarity_search(query, k=fetch_k)

    filtered = [
        doc for doc in candidates
        if all(doc.metadata.get(key) == val for key, val in filter_dict.items())
    ]
    return filtered[:k]


# =============================================================================
# 메인
# =============================================================================

parser = argparse.ArgumentParser(description="Embedding experiment (Chroma / FAISS)")
parser.add_argument("--config", default="configs/experiments/bge-m3_qwen3-8B.yaml")
args = parser.parse_args()

config = load_config(args.config)

# vector_db 타입 결정: YAML의 retrieval.vector_db 키 우선, 없으면 "chroma"
vector_db_type = config["retrieval"].get("vector_db", "chroma").lower()

print(
    "config loaded "
    f"(vector_db: {vector_db_type}, "
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

    for local_chunk_id, chunk in enumerate(chunks):
        global_chunk_id = len(documents)
        metadata = {
            "document_id": document_id,
            "source": project_name,
            "source_file": pdf_path,
            "chunk_id": global_chunk_id,
            "local_chunk_id": local_chunk_id,
        }
        documents.append(Document(page_content=chunk, metadata=metadata))
        chunk_records.append(
            {
                **metadata,
                "chunking_strategy": splitter,
                "chunk_length": len(chunk),
                "chunk_text": chunk,
            }
        )

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
# 3. 벡터 DB 빌드 — Chroma / FAISS 분기
#    build_seconds: from_documents() + save 전체를 동일 기준으로 측정
#    peak_ram_mb: 빌드 전후 프로세스 RSS 차이
# =============================================================================

persist_db_b = resolve_project_path(config["retrieval"]["persist_directory"])
os.makedirs(persist_db_b, exist_ok=True)

process = psutil.Process(os.getpid())
ram_before_mb = process.memory_info().rss / (1024 ** 2)

start_db = time.time()

if vector_db_type == "faiss":
    vector_db_b = FAISS.from_documents(documents=documents, embedding=embeddings_b)
    # FAISS는 save_local까지 포함해 build_seconds를 측정 (Chroma와 동일 기준)
    vector_db_b.save_local(persist_db_b)

else:  # chroma (default)
    vector_db_b = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_b,
        persist_directory=persist_db_b,
    )

embedding_build_seconds = time.time() - start_db
ram_after_mb = process.memory_info().rss / (1024 ** 2)
peak_ram_delta_mb = max(ram_after_mb - ram_before_mb, 0.0)

# 인덱스 파일 크기 (디스크)
index_size_mb = get_dir_size_mb(persist_db_b)

embedding_build_seconds_per_chunk = (
    embedding_build_seconds / len(documents) if documents else 0.0
)

print(f"Vector DB built and saved ({embedding_build_seconds:.2f}s): {persist_db_b}")
print(f"  Index size on disk : {index_size_mb:.2f} MB")
print(f"  RAM delta (build)  : {peak_ram_delta_mb:.2f} MB")

# =============================================================================
# 4. 샘플 쿼리 테스트
# =============================================================================

query = config["retrieval"]["sample_query"]
print(f"\n[Sample query test] Question: '{query}'")
sample_document_id = config["retrieval"].get("sample_document_id", document_configs[0]["document_id"])

if vector_db_type == "faiss":
    retrieved_docs_b = faiss_similarity_search_with_filter(
        vector_db_b, query, k=2, filter_dict={"document_id": sample_document_id}
    )
else:
    retrieved_docs_b = vector_db_b.similarity_search(
        query, k=2, filter={"document_id": sample_document_id}
    )

print("=" * 50)
for idx, doc in enumerate(retrieved_docs_b):
    print(f"[Retrieved chunk {idx + 1}] (Chunk ID: {doc.metadata['chunk_id']})")
    print(doc.page_content)
    print("-" * 50)

# =============================================================================
# 5. Ground Truth 기반 Retrieval 평가
#    query_latency_ms: 질문별 검색 지연시간 (ms)
# =============================================================================

retrieval_k = int(os.environ.get("RAG_RETRIEVAL_K", str(config["retrieval"].get("top_k", 3))))
strategy_name = config["output"]["strategy_name"]

ground_truth_df = make_ground_truth_dataframe(
    [document_config["document_id"] for document_config in document_configs]
)
retrieval_rows = []

for _, row in ground_truth_df.iterrows():
    question = row["question"]
    document_filter = {"document_id": row["document_id"]}

    start_q = time.time()
    if vector_db_type == "faiss":
        retrieved_docs = faiss_similarity_search_with_filter(
            vector_db_b, question, k=retrieval_k, filter_dict=document_filter
        )
    else:
        retrieved_docs = vector_db_b.similarity_search(
            question, k=retrieval_k, filter=document_filter
        )
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
        }
    )

retrieval_df = pd.DataFrame(retrieval_rows)
evaluated_df = evaluate_retrieval_dataframe(retrieval_df)
summary_df = summarize_retrieval(evaluated_df)

# =============================================================================
# 6. 결과 저장
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
