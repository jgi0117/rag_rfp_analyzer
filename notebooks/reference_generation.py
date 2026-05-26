import argparse
import json
import os
import sys

import pandas as pd
import yaml
from dotenv import load_dotenv
import torch
from transformers import pipeline as hf_pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
parser = argparse.ArgumentParser(description="Reference generation evaluation experiment")
parser.add_argument("--config", default="configs/experiments/reference.yaml")
args = parser.parse_args()

config = load_config(args.config)
print(
    "config loaded "
    f"(db: {config['retrieval']['persist_directory']}, "
    f"embedding: {config['embedding']['model']}, "
    f"generation: {config['generation']['model']})"
)

load_dotenv()

# 1. embedding 단계에서 생성된 벡터 DB 로드
embed_provider = config["embedding"]["provider"]
embed_model    = config["embedding"]["model"]
embed_model_name = f"{embed_provider}/{embed_model}" if "/" not in embed_model else embed_model
print(f"임베딩 모델 로드 중: {embed_model_name}")
embeddings_b = HuggingFaceEmbeddings(model_name=embed_model_name)
persist_db_b = resolve_project_path(config["retrieval"]["persist_directory"])
if not os.path.exists(persist_db_b):
    raise FileNotFoundError(
        f"Chroma DB directory not found: {persist_db_b}\n"
        "먼저 reference_embedding.py를 실행해 DB를 생성해 주세요."
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
    retrieved_docs = vector_db_b.similarity_search(question, k=retrieval_k, filter=document_filter)
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

gen_provider   = config["generation"]["provider"]
gen_model      = config["generation"]["model"]
gen_model_name = f"{gen_provider}/{gen_model}" if "/" not in gen_model else gen_model
gen_temperature = config["generation"].get("temperature", 0.0)
print(f"생성 모델 로드 중: {gen_model_name}")

gen_pipe = hf_pipeline(
    "text-generation",
    model=gen_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=512,
    do_sample=gen_temperature > 0,
    temperature=gen_temperature if gen_temperature > 0 else None,
    return_full_text=False,
    trust_remote_code=True,
)

# 모델 로드 완료 후 sys.modules에 올라온 modeling_exaone 모듈에서
# create_causal_mask를 직접 교체 — 최신 transformers가 넘기는 input_embeds 인자를 무시
import inspect as _inspect
for _mod_name, _mod in list(sys.modules.items()):
    if "exaone" in _mod_name.lower() and hasattr(_mod, "create_causal_mask"):
        _orig_ccm = _mod.create_causal_mask
        _valid_params = set(_inspect.signature(_orig_ccm).parameters.keys())
        def _patched_ccm(*args, _orig=_orig_ccm, _valid=_valid_params, **kwargs):
            # 최신 transformers: input_embeds (s 없음) → EXAONE: inputs_embeds (s 있음)
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            filtered = {k: v for k, v in kwargs.items() if k in _valid}
            return _orig(*args, **filtered)
        _mod.create_causal_mask = _patched_ccm
        print(f"[compat] create_causal_mask 패치 완료 ({_mod_name}), 허용 파라미터: {_valid_params}")
        break
else:
    print("[compat] exaone 모듈을 sys.modules에서 찾지 못함 — 패치 스킵")


def call_llm(prompt: str) -> str:
    result = gen_pipe(prompt)
    return result[0]["generated_text"]

# 검색 컨텍스트 정보를 취합하여 실제 모델 답변을 생성합니다.
for _, row in evaluated_df.iterrows():
    question = row["question"]
    joined_contexts = "\n\n".join(row["retrieved_contexts"])

    qa_prompt = (
        "당신은 제안서 분석 전문가입니다. 주어진 문맥에만 철저히 기반하여 질문에 답하세요.\n"
        "문맥에 없는 내용이거나 확인 불가능한 정보라면 솔직하게 '문맥상 확인할 수 없습니다'라고 답하세요.\n\n"
        f"[문맥]:\n{joined_contexts}\n\n[질문]:\n{question}"
    )
    llm_answer = call_llm(qa_prompt)
    generated_answers.append(llm_answer)

# 공통 평가지표 함수 입력을 위한 필수 컬럼 정제
evaluated_df["generated_answer"] = generated_answers
evaluated_df["retrieved_context"] = evaluated_df["retrieved_contexts"].apply(lambda x: "\n\n".join(x))


# 3. src/evaluation/generation.py 내 공통 인터페이스 호출
def openai_judge_fn(prompt: str) -> str:
    return call_llm(prompt)


final_generation_df = evaluate_generation_dataframe(
    evaluated_df,
    question_col="question",
    ground_truth_col="ground_truth_answer",
    context_col="retrieved_context",
    answer_col="generated_answer",
    judge_fn=openai_judge_fn,
)

# 전략별 종합 평균 요약 계산
generation_summary_df = summarize_by_strategy(final_generation_df, strategy_col="strategy")

# 4. 최종 결과물 로컬 디렉토리 저장
generation_output_path = resolve_project_path(
    config["output"]["generation_eval_results"]
)
generation_summary_path = resolve_project_path(
    config["output"]["generation_eval_summary"]
)

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
