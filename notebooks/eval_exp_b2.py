import sys
import os
import time
import yaml
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# =========================================================
# 1. 경로 설정 및 팀원 모듈 임포트
# =========================================================
current_file_path = os.path.abspath(__file__)
root_name = "rag_rfp_analyzer"

if root_name in current_file_path:
    project_root_dir = current_file_path.split(root_name)[0] + root_name
else:
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(current_file_path), ".."))

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

print(f"🚀 탐색된 프로젝트 절대 경로: {project_root_dir}")

from src.preprocessing.loader import extract_pdf
from src.evaluation.retrieval import (
    make_default_ground_truth_dataframe, 
    evaluate_retrieval_dataframe, 
    summarize_retrieval
)

load_dotenv()

# =========================================================
# 2. 고정 크기 청킹 (Fixed-size Chunking) 및 벡터 DB 구축
# =========================================================
config_path = os.path.join(project_root_dir, "config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

yaml_file_dir = config['path']['file_dir'].replace("../", "")
filepath = os.path.join(project_root_dir, yaml_file_dir, "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf")
raw_text = extract_pdf(filepath, pages=None)

# 의미 단위 헤더 분할 없이, 텍스트 전체를 바로 고정 규격으로 쪼갭니다.
fixed_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

splitted_texts = fixed_splitter.split_text(raw_text)

final_documents = []
for chunk_idx, text in enumerate(splitted_texts):
    meta = {
        "source": "고려대학교_RFP_고정청킹",
        "chunk_id": int(chunk_idx)
    }
    final_documents.append(Document(page_content=text, metadata=meta))

print(f"📊 [고정 청킹] 총 생성된 청크 수: {len(final_documents)}개")

print("📥 OpenAI Embedding API 연결 및 벡터 DB 구축 중...")
embeddings_fixed = OpenAIEmbeddings(model="text-embedding-3-small")
persist_db_fixed = os.path.join(project_root_dir, "chroma_db_fixed")

# 중복 구축 방지를 위해 기존 DB 폴더가 있다면 초기화
if os.path.exists(persist_db_fixed):
    import shutil
    shutil.rmtree(persist_db_fixed)

vector_db = Chroma.from_documents(
    documents=final_documents,
    embedding=embeddings_fixed,
    persist_directory=persist_db_fixed
)
print("✅ 고정 청킹 벡터 DB 구축 완료!")


# =========================================================
# 3. 팀원 평가 표준 템플릿(10개 문항) 기반 실시간 검색 및 본문 매핑
# =========================================================
print("\n🧪 [통합 평가] 팀원 표준 템플릿 기반 실시간 Retrieval 구동...")
eval_df = make_default_ground_truth_dataframe() 

retrieved_results = []
for idx, row in eval_df.iterrows():
    query = row['question']
    retrieved_docs = vector_db.similarity_search(query, k=4) # 대조군 실험을 위해 k=4 동일하게 유지
    
    contexts_list = [doc.page_content for doc in retrieved_docs]
    retrieved_results.append(contexts_list)

eval_df['retrieved_contexts'] = retrieved_results
eval_df['strategy'] = "고정 크기 일반 Chunking"


# =========================================================
# 4. 종합 점수 일괄 계산 및 리포트 출력
# =========================================================
report_df = evaluate_retrieval_dataframe(
    eval_df,
    question_col="question",
    ground_truth_col="ground_truth",
    contexts_col="retrieved_contexts",
    relevance_threshold=0.2, 
    ndcg_k=4
)

summary_df = summarize_retrieval(report_df, group_col="strategy")

print("\n" + "="*80)
print("📊 [고정 크기 청킹 + 팀원 통합 평가 모듈 실시간 정량 지표]")
print("="*80)
print(report_df[['question_id', 'context_recall', 'context_precision', 'mrr', 'ndcg', 'chunk_relevance_scores']].to_markdown(index=False))

print("\n🏆 [전체 평균 요약 스코어]")
print(summary_df.to_markdown(index=False))