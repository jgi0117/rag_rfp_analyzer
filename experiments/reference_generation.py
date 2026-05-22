import argparse
import json
import os
import sys
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
sys.path.append(project_root_dir)

# 🌟 기존 Retrieval 평가 함수들과 새로 매핑할 Generation 평가 공통 함수들을 각각 임포트합니다.
from src.evaluation.retrieval import (
    evaluate_retrieval_dataframe,
    make_default_ground_truth_dataframe,
    summarize_retrieval,
)
from src.evaluation.generation import evaluate_generation_dataframe, summarize_by_strategy
from src.preprocessing.cleaner import RFPTextCleaner
from src.preprocessing.loader import extract_pdf


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


# 본 스크립트는 시나리오 B fixed-size 청킹 기반의 'Generation 통합 평가'를 수행합니다.
parser = argparse.ArgumentParser(description="Scenario B fixed-size generation evaluation experiment")
parser.add_argument("--config", default="configs/experiments/fixed_b.yaml")
args = parser.parse_args()

config = load_config(args.config)
print(
    "config loaded "
    f"(splitter: {config['preprocessing']['splitter']}, "
    f"chunk_size: {config['preprocessing']['chunk_size']}, "
    f"overlap: {config['preprocessing']['chunk_overlap']})"
)

# 1. 문서 전처리 및 청킹 로드 (reference_embedding.py 복사본 흐름 보존)
pdf_path = resolve_project_path(config["path"]["raw_pdf_file"])
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

md_text = extract_pdf(
    pdf_path,
    pages=config["path"].get("pdf_pages"),
    image_path=resolve_project_path(config["path"].get("image_dir", "outputs/images")),
    write_images=config["path"].get("write_images", False),
)

project_name = config["project"]["target_document"]
cleaner = RFPTextCleaner(config=config)
chunks = cleaner.run_fixed_size_chunking(md_text, project_name=project_name)
print(f"Total chunks: {len(chunks)}")

documents = [
    Document(page_content=chunk, metadata={"source": project_name, "chunk_id": i})
    for i, chunk in enumerate(chunks)
]

# 2. 임베딩 벡터 데이터베이스 구축 및 타겟 쿼리 테스트
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

embeddings_b = OpenAIEmbeddings(model=config["embedding"]["model"])
persist_db_b = resolve_project_path(config["retrieval"]["persist_directory"])

start_db = time.time()
vector_db_b = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_b,
    persist_directory=persist_db_b,
)
print(f"Vector DB built and saved ({time.time() - start_db:.2f}s): {persist_db_b}")

# 3. Ground Truth 기반의 Retrieval 검색 데이터 수집
retrieval_k = int(os.environ.get("RAG_RETRIEVAL_K", str(config["retrieval"].get("top_k", 3))))
strategy_name = config["output"]["strategy_name"]

ground_truth_df = make_default_ground_truth_dataframe()
retrieval_rows = []

for _, row in ground_truth_df.iterrows():
    question = row["question"]
    retrieved_docs = vector_db_b.similarity_search(question, k=retrieval_k)
    retrieved_ranked_chunks = [
        {
            "rank": rank,
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "source": doc.metadata.get("source", ""),
            "chunk_text": doc.page_content,
        }
        for rank, doc in enumerate(retrieved_docs, start=1)
    ]
    retrieval_rows.append(
        {
            "strategy": strategy_name,
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

# ==============================================================================
# 🔥 [Generation 핵심 수행 영역] 
# ==============================================================================
print("\n🤖 [Generation 파이프라인 진입] 각 문항별 RAG 답변 생성 중...")

# 공통 모듈 가동을 위해 기존 retrieval 규격 컬럼을 generation_eval_metrics 규격으로 변환
evaluated_df = evaluated_df.rename(columns={"ground_truth": "ground_truth_answer"})
generated_answers = []

generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# 검색 컨텍스트 정보를 취합하여 실제 모델 답변을 생성합니다.
for idx, row in evaluated_df.iterrows():
    question = row["question"]
    joined_contexts = "\n\n".join(row["retrieved_contexts"])
    
    qa_prompt = (
        f"당신은 제안서 분석 전문가입니다. 주어진 문맥에만 철저히 기반하여 질문에 답하세요.\n"
        f"문맥에 없는 내용이거나 확인 불가능한 정보라면 솔직하게 '문맥상 확인할 수 없습니다'라고 답하세요.\n\n"
        f"[문맥]:\n{joined_contexts}\n\n[질문]:\n{question}"
    )
    llm_answer = generator_llm.predict(qa_prompt)
    generated_answers.append(llm_answer)

# 공통 평가지표 함수 입력을 위한 필수 컬럼 정제
evaluated_df["generated_answer"] = generated_answers
evaluated_df["retrieved_context"] = evaluated_df["retrieved_contexts"].apply(lambda x: "\n\n".join(x))


# 4. src/evaluation/generation.py 내 공통 인터페이스 호출 (LLM Judge 연동)
def openai_judge_fn(prompt: str) -> str:
    return generator_llm.predict(prompt)

final_generation_df = evaluate_generation_dataframe(
    evaluated_df,
    question_col="question",
    ground_truth_col="ground_truth_answer",
    context_col="retrieved_context",
    answer_col="generated_answer",
    judge_fn=openai_judge_fn
)

# 전략별 종합 평균 요약 계산
generation_summary_df = summarize_by_strategy(final_generation_df, strategy_col="strategy")

# 5. 최종 결과물 로컬 디렉토리 저장 조치 (출력 경로 내 retrieval 단어를 generation으로 전환)
generation_output_path = config["output"]["retrieval_eval_results"].replace("retrieval", "generation")
generation_output_path = resolve_project_path(generation_output_path)
generation_summary_path = config["output"]["retrieval_eval_summary"].replace("retrieval", "generation")
generation_summary_path = resolve_project_path(generation_summary_path)

os.makedirs(os.path.dirname(generation_output_path), exist_ok=True)
final_generation_df.to_csv(generation_output_path, index=False, encoding="utf-8-sig")
generation_summary_df.to_csv(generation_summary_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("📊 [RAG 파이프라인 통합 평가 완료 리포트 (Generation 요약)]")
print("=" * 80)
print(generation_summary_df.to_string(index=False))
print("=" * 80)
print(f"✅ 문항별 상세 평가 데이터 저장 완료: {generation_output_path}")
print(f"✅ 전략별 평균 요약 데이터 저장 완료: {generation_summary_path}")