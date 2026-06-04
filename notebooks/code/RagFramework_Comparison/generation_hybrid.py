import argparse
import json
import os
import sys
import time

import pandas as pd
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

from src.evaluation.generation import evaluate_generation_dataframe, summarize_by_strategy
from src.evaluation.ground_truth import make_ground_truth_dataframe
from src.evaluation.retrieval import (
    evaluate_retrieval_dataframe,
    summarize_retrieval,
)

# PDF 재파싱 안 하니까 제거
# from src.preprocessing.cleaner import RFPTextCleaner
# from src.preprocessing.loader import extract_pdf


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


def run_chunking(cleaner, splitter: str, md_text: str, project_name: str) -> list[str]:
    if splitter == "markdown":
        return cleaner.run_markdown_chunking(md_text, project_name=project_name)
    raise ValueError("Baseline only supports markdown chunking.")


def get_document_ids(config: dict) -> list[str]:
    documents = config.get("documents")
    if documents:
        return [doc["document_id"] for doc in documents]
    return ["korea_portal"]


def build_hybrid_retrievers(
    vector_db: Chroma,
    docs_by_document_id: dict[str, list[Document]],
    retrieval_k: int,
    dense_weight: float = 0.5,
) -> dict[str, EnsembleRetriever]:
    """
    document_id별로 EnsembleRetriever를 생성해 반환한다.
    embedding_hybrid.py와 동일한 구성 — 동작 기준 통일.
    """
    sparse_weight = 1.0 - dense_weight
    retrievers: dict[str, EnsembleRetriever] = {}

    for document_id, docs in docs_by_document_id.items():
        chroma_retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": retrieval_k,
                "filter": {"document_id": document_id},
            },
        )

        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = retrieval_k

        ensemble = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25_retriever],
            weights=[dense_weight, sparse_weight],
        )
        retrievers[document_id] = ensemble

    return retrievers


# =============================================================================
# 메인
# 본 스크립트는 embedding_hybrid.py에서 저장한 Chroma DB를 로드하고,
# BM25 인덱스를 재구성해 Hybrid Retrieval + Generation 평가를 수행합니다.
# BM25는 인메모리 인덱스라 매번 청크에서 재구성합니다.
# embedding 단계에서 이미 PDF 파싱 + 청킹을 했는데, generation_hybrid.py가 BM25 재구성한다고 PDF를 또 파싱하는 것을 막기 위해
# =============================================================================

parser = argparse.ArgumentParser(description="LangChain Hybrid generation experiment")
parser.add_argument("--config", default="configs/experiments/bge-m3_qwen3-8B.yaml")
args = parser.parse_args()

config = load_config(args.config)
dense_weight = float(config["retrieval"].get("dense_weight", 0.5))

print(
    "config loaded "
    f"(framework: langchain_hybrid, "
    f"dense_weight: {dense_weight}, sparse_weight: {1.0 - dense_weight:.1f}, "
    f"db: {config['retrieval']['persist_directory']}, "
    f"embedding: {config['embedding']['model']}, "
    f"generation: {config['generation']['model']})"
)

# =============================================================================
# 1. 임베딩 모델 로드
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
# 2. Chroma DB 로드
# =============================================================================

persist_db_b = resolve_project_path(config["retrieval"]["persist_directory"])
if not os.path.exists(persist_db_b):
    raise FileNotFoundError(
        f"Chroma DB directory not found: {persist_db_b}\n"
        "먼저 embedding_hybrid.py를 실행해 DB를 생성해 주세요."
    )

vector_db = Chroma(
    persist_directory=persist_db_b,
    embedding_function=embeddings_b,
)
print(f"Vector DB (Chroma/Dense) loaded: {persist_db_b}")

# =============================================================================
# 3. BM25 인덱스 재구성을 위한 청크 로드
#    BM25는 인메모리라 저장이 불가 — embedding 단계와 동일한 청킹으로 재구성
#  embedding 단계에서 저장한 chunk CSV를 generation에서 그대로 읽으면 PDF 재파싱이 필요 없습니다.
# =============================================================================

print("BM25 인덱스 재구성을 위해 chunk CSV 로드 중...")

chunk_csv_path = resolve_project_path(config["output"]["chunk_results"])
if not os.path.exists(chunk_csv_path):
    raise FileNotFoundError(
        f"Chunk CSV not found: {chunk_csv_path}\n"
        "먼저 embedding_hybrid.py를 실행해 주세요."
    )

chunk_df = pd.read_csv(chunk_csv_path, encoding="utf-8-sig")
docs_by_document_id: dict[str, list[Document]] = {}

for _, r in chunk_df.iterrows():
    doc = Document(
        page_content=r["chunk_text"],
        metadata={
            "document_id": r["document_id"],
            "source": r["source"],
            "source_file": r["source_file"],
            "chunk_id": int(r["chunk_id"]),
            "local_chunk_id": int(r["local_chunk_id"]),
        },
    )
    docs_by_document_id.setdefault(r["document_id"], []).append(doc)

for doc_id, docs in docs_by_document_id.items():
    print(f"  로드 완료 ({doc_id}): {len(docs)} chunks")

# =============================================================================
# 4. Hybrid Retriever 빌드
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
# 5. Ground Truth 기반 Retrieval 검색
# =============================================================================

strategy_name = config["output"]["strategy_name"]
ground_truth_df = make_ground_truth_dataframe(get_document_ids(config))
retrieval_rows = []

for _, row in ground_truth_df.iterrows():
    question = row["question"]
    document_id = row["document_id"]

    retriever = hybrid_retrievers[document_id]
    retrieved_docs = retriever.invoke(question)[:retrieval_k]

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
        }
    )

retrieval_df = pd.DataFrame(retrieval_rows)
evaluated_df = evaluate_retrieval_dataframe(retrieval_df)
_retrieval_summary_df = summarize_retrieval(evaluated_df)

# =============================================================================
# 6. Generation
# =============================================================================

print("\n[Generation 파이프라인 진입] 각 문항별 RAG 답변 생성 중...")

evaluated_df = evaluated_df.rename(columns={"ground_truth": "ground_truth_answer"})
generated_answers = []
generation_seconds = []

generation_provider = config["generation"]["provider"]

if generation_provider == "openai":
    from langchain_openai import ChatOpenAI

    generator_llm = ChatOpenAI(
        model=config["generation"]["model"],
        temperature=config["generation"].get("temperature", 0.0),
    )

    def call_llm(prompt: str) -> str:
        response = generator_llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

elif generation_provider == "huggingface":
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    model_name = config["generation"]["model"]

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("is_quantized:", qwen_model.is_quantized)
    for name, param in list(qwen_model.named_parameters())[:5]:
        print(f"{name}: {param.dtype}")
    print(f"VRAM 사용: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def call_llm(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = qwen_tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(qwen_model.device)

        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=config["generation"].get("max_new_tokens", 512),
                do_sample=False,
            )

        return qwen_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

else:
    raise ValueError(f"Unknown generation provider: {generation_provider}")

for idx, (_, row) in enumerate(evaluated_df.iterrows()):
    question = row["question"]
    joined_contexts = "\n\n".join(row["retrieved_contexts"])

    qa_prompt = (
        "당신은 제안서 분석 전문가입니다. 주어진 문맥에만 철저히 기반하여 질문에 답하세요.\n"
        "문맥에 없는 내용이거나 확인 불가능한 정보라면 솔직하게 '문맥상 확인할 수 없습니다'라고 답하세요.\n\n"
        f"[문맥]:\n{joined_contexts}\n\n[질문]:\n{question}"
    )

    generation_start = time.time()
    llm_answer = call_llm(qa_prompt)
    elapsed_generation_seconds = time.time() - generation_start

    generated_answers.append(llm_answer)
    generation_seconds.append(elapsed_generation_seconds)

    warmup_tag = " [warmup — GPU 첫 실행, 시간 참고용]" if idx == 0 else ""
    print(
        f"Generated answer "
        f"(document_id={row['document_id']}, question_id={row['question_id']}, "
        f"{elapsed_generation_seconds:.2f}s){warmup_tag}"
    )

evaluated_df["generated_answer"] = generated_answers
evaluated_df["generation_seconds"] = generation_seconds
evaluated_df["retrieved_context"] = evaluated_df["retrieved_contexts"].apply(lambda x: "\n\n".join(x))

# =============================================================================
# 7. Generation 평가 (judge)
# =============================================================================

judge_fn = None
judge_model = config["generation"].get("judge_model")
if judge_model:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. "
            "Skipping OpenAI judge and using heuristic generation metrics."
        )
    else:
        from langchain_openai import ChatOpenAI

        judge_llm = ChatOpenAI(
            model=judge_model,
            temperature=0.0,
        )

        def openai_judge_fn(prompt: str) -> str:
            response = judge_llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        judge_fn = openai_judge_fn

final_generation_df = evaluate_generation_dataframe(
    evaluated_df,
    question_col="question",
    ground_truth_col="ground_truth_answer",
    context_col="retrieved_context",
    answer_col="generated_answer",
    judge_fn=judge_fn,
)

generation_summary_df = summarize_by_strategy(final_generation_df, strategy_col="strategy")

# =============================================================================
# 8. 결과 저장
# =============================================================================

generation_output_path = resolve_project_path(config["output"]["generation_eval_results"])
generation_summary_path = resolve_project_path(config["output"]["generation_eval_summary"])

os.makedirs(os.path.dirname(generation_output_path), exist_ok=True)
final_generation_df.to_csv(generation_output_path, index=False, encoding="utf-8-sig")
generation_summary_df.to_csv(generation_summary_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("[RAG 파이프라인 통합 평가 완료 리포트 (Generation 요약)]")
print("=" * 80)
print(generation_summary_df.to_string(index=False))
print("=" * 80)
print(f"문항별 상세 평가 데이터 저장 완료: {generation_output_path}")
print(f"전략별 평균 요약 데이터 저장 완료: {generation_summary_path}")
