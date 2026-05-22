import sys
import os
import time
import yaml
import pandas as pd
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# =========================================================
# 1. 경로 설정 및 모듈 임포트
# =========================================================
# 현재 파일(embedding_exp.py)의 위치를 기준으로 프로젝트 루트 경로를 계산하여 추가합니다.
current_file_dir = os.path.dirname(os.path.abspath(__file__)) # notebooks/ 폴더 위치
project_root_dir = os.path.abspath(os.path.join(current_file_dir, "..")) # project_root/ 폴더 위치
sys.path.append(project_root_dir)

from src.preprocessing.cleaner import RFPTextCleaner
from src.evaluation.retrieval import (
    evaluate_retrieval_dataframe,
    make_default_ground_truth_dataframe,
    summarize_retrieval,
)


# =========================================================
# 2. config.yaml 설정 파일 로드 및 실험 조건 적용
# =========================================================
# 터미널 실행 위치에 상관없이 프로젝트 루트의 config.yaml을 정확히 찾아옵니다.
config_path = os.path.join(project_root_dir, "config.yaml")

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 💡 config.yaml을 직접 수정하셨더라도, 혹시 모를 오차를 방지하기 위해 
# 내 실험 조건(500자, 오버랩 0)을 코드 상에서 한 번 더 확실하게 고정해 줍니다.
config['preprocessing']['chunk_size'] = 500
config['preprocessing']['chunk_overlap'] = 0

print(f"⚙️ config.yaml 로드 완료! (실험 조건 -> Chunk Size: {config['preprocessing']['chunk_size']})")


def resolve_project_path(path_value: str) -> str:
    path_value = os.path.expanduser(path_value)
    if os.path.isabs(path_value):
        return path_value

    parts = path_value.replace("\\", "/").split("/")
    if parts and parts[0] == os.path.basename(project_root_dir):
        path_value = "/".join(parts[1:])
    return os.path.join(project_root_dir, path_value)


# =========================================================
# 3. 데이터 로드 및 고정 크기 청킹 가동
# =========================================================
# config에 적힌 단일 고려대학교 Markdown 경로를 시스템 절대 경로로 결합하여 에러를 방지합니다.
markdown_path = resolve_project_path(config["path"]["markdown_file"])
if not os.path.exists(markdown_path):
    raise FileNotFoundError(f"Markdown 파일을 찾을 수 없습니다: {markdown_path}")

# 저장된 마크다운 텍스트 로드 및 예원님 전용 고정 크기 청킹 실행
with open(markdown_path, "r", encoding="utf-8-sig") as f:
    md_text = f.read()

cleaner = RFPTextCleaner(config=config)
pure_python_chunks = cleaner.run_fixed_size_chunking(md_text, project_name="고려대_차세대포털")
print(f"📊 총 생성된 청크 수: {len(pure_python_chunks)}개")


# =========================================================
# 4. 랭체인 Document 객체 형식으로 래핑
# =========================================================
documents = [
    Document(page_content=chunk, metadata={"source": "고려대학교_RFP", "chunk_id": i})
    for i, chunk in enumerate(pure_python_chunks)
]


# =========================================================
# 5. [시나리오 B 실험] .env 보안 키 로드 및 OpenAI 임베딩 DB 구축
# =========================================================

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("🚨 OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")

print("📥 OpenAI 임베딩 모델 연결 중...")
embeddings_b = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma DB 저장 경로도 절대 경로로 지정하여 유실을 막습니다.
persist_db_b = os.path.join(project_root_dir, "chroma_db_scenario_b")

start_db = time.time()
vector_db_b = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_b,
    persist_directory=persist_db_b
)
print(f"✅ [시나리오 B] 벡터 DB 구축 및 저장 완료! (소요 시간: {time.time() - start_db:.2f}초)")



# =========================================================
# 6. 간단한 검색 테스트 검증
# =========================================================

query = "서울캠퍼스와 세종캠퍼스의 교직원 현황 및 전임교원 수는 어떻게 되나요?"
print(f"\n🔍 [시나리오 B 테스트] 질문: '{query}'")
retrieved_docs_b = vector_db_b.similarity_search(query, k=2)

print("="*50)
for idx, doc in enumerate(retrieved_docs_b):
    print(f"🌟 [검색된 청크 {idx+1}] (원본 Chunk ID: {doc.metadata['chunk_id']})")
    print(doc.page_content)
    print("-" * 50)


# =========================================================
# 7. retrieval 평가 코드 연결
# =========================================================
retrieval_k = int(os.environ.get("RAG_RETRIEVAL_K", str(config.get("retrieval", {}).get("top_k", 3))))
strategy_name = "scenario_b_openai_fixed_500_overlap_0"

ground_truth_df = make_default_ground_truth_dataframe()
retrieval_rows = []

for _, row in ground_truth_df.iterrows():
    question = row["question"]
    retrieved_docs = vector_db_b.similarity_search(question, k=retrieval_k)
    retrieval_rows.append(
        {
            "strategy": strategy_name,
            "question_id": row["question_id"],
            "question": question,
            "ground_truth": row["ground_truth"],
            "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
            "retrieved_chunk_ids": ", ".join(str(doc.metadata.get("chunk_id", "")) for doc in retrieved_docs),
        }
    )

retrieval_df = pd.DataFrame(retrieval_rows)
evaluated_df = evaluate_retrieval_dataframe(retrieval_df)
summary_df = summarize_retrieval(evaluated_df)

output_dir = resolve_project_path(config.get("output", {}).get("evaluation_dir", "outputs/evaluation"))
os.makedirs(output_dir, exist_ok=True)

retrieval_output_path = os.path.join(output_dir, "scenario_b_retrieval_eval_results.csv")
summary_output_path = os.path.join(output_dir, "scenario_b_retrieval_eval_summary.csv")

evaluated_df.to_csv(retrieval_output_path, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_output_path, index=False, encoding="utf-8-sig")

print("\n✅ [시나리오 B] retrieval 평가 완료!")
print(f"📊 평가 결과 저장: {retrieval_output_path}")
print(f"📈 요약 결과 저장: {summary_output_path}")
print("\n[retrieval 평가 요약]")
print(summary_df.to_string(index=False))
