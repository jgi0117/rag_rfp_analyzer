import argparse
import json
import os
import sys
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
sys.path.append(project_root_dir)

from src.evaluation.generation import evaluate_generation_dataframe, summarize_by_strategy
from src.evaluation.ground_truth import make_ground_truth_dataframe
from src.evaluation.retrieval import (
    evaluate_retrieval_dataframe,
    summarize_retrieval,
)


def resolve_project_path(path_value: str) -> str:
    path_value = os.path.expanduser(path_value)
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(project_root_dir, path_value)


def get_persist_directory(config: dict) -> str:
    persist_directory = config.get("retrieval", {}).get("persist_directory")
    if persist_directory:
        return resolve_project_path(persist_directory)
    return resolve_project_path(os.path.join("outputs", "db", config["output"]["strategy_name"]))


def resolve_output_path(config: dict, key: str) -> str:
    path_value = config["output"][key].format(
        strategy_name=config["output"]["strategy_name"]
    )
    return resolve_project_path(path_value)


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


def get_document_ids(config: dict) -> list[str]:
    documents = config.get("documents")
    if documents:
        return [document["document_id"] for document in documents]
    return ["korea_portal"]


# 본 스크립트는 embedding 단계에서 생성된 Chroma DB를 로드해 Generation 통합 평가를 수행합니다.
parser = argparse.ArgumentParser(description="Baseline generation evaluation experiment")
parser.add_argument("--config", default="configs/experiments/bge-m3_qwen3-8B.yaml")
args = parser.parse_args()

config = load_config(args.config)
persist_db_b = get_persist_directory(config)
print(
    "config loaded "
    f"(db: {persist_db_b}, "
    f"embedding: {config['embedding']['model']}, "
    f"generation: {config['generation']['model']})"
)

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

if not os.path.exists(persist_db_b):
    raise FileNotFoundError(
        f"Chroma DB directory not found: {persist_db_b}\n"
        "먼저 baseline_embedding.py를 실행해 DB를 생성해 주세요."
    )

vector_db_b = Chroma(
    persist_directory=persist_db_b,
    embedding_function=embeddings_b,
)
print(f"Vector DB loaded: {persist_db_b}")

# 2. Ground Truth 기반의 Retrieval 검색 데이터 수집
retrieval_k = int(os.environ.get("RAG_RETRIEVAL_K", str(config["retrieval"].get("top_k", 4))))
strategy_name = config["output"]["strategy_name"]

ground_truth_df = make_ground_truth_dataframe(get_document_ids(config))
retrieval_rows = []

for _, row in ground_truth_df.iterrows():
    question = row["question"]
    document_filter = {"document_id": row["document_id"]}

    retrieved_docs = vector_db_b.similarity_search(
        question,
        k=retrieval_k,
        filter=document_filter,
    )
    
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

# ==============================================================================
# [Generation 핵심 수행 영역]
# ==============================================================================
print("\n[Generation 파이프라인 진입] 각 문항별 RAG 답변 생성 중...")

# 공통 모듈 가동을 위해 기존 retrieval 규격 컬럼을 generation_eval_metrics 규격으로 변환
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_name = config["generation"]["model"]
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )

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

# 검색 컨텍스트 정보를 취합하여 실제 모델 답변을 생성합니다.
for _, row in evaluated_df.iterrows():
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
    print(
        f"Generated answer "
        f"(document_id={row['document_id']}, question_id={row['question_id']}, "
        f"{elapsed_generation_seconds:.2f}s)"
    )

# 공통 평가지표 함수 입력을 위한 필수 컬럼 정제
evaluated_df["generated_answer"] = generated_answers
evaluated_df["generation_seconds"] = generation_seconds
evaluated_df["retrieved_context"] = evaluated_df["retrieved_contexts"].apply(lambda x: "\n\n".join(x))


# 3. src/evaluation/generation.py 내 공통 인터페이스 호출
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

# 전략별 종합 평균 요약 계산
generation_summary_df = summarize_by_strategy(final_generation_df, strategy_col="strategy")

# 4. 최종 결과물 로컬 디렉토리 저장
generation_output_path = resolve_output_path(config, "generation_eval_results")
generation_summary_path = resolve_output_path(config, "generation_eval_summary")

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
