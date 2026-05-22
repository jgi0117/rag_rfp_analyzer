import sys
import os
import time
import yaml
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# =========================================================
# 1. 경로 설정 및 팀원 모듈 임포트 (경로 에러 완벽 방어 버전)
# =========================================================
# 현재 실행 중인 파일(eval_exp_semantic_b2.py)의 절대 경로를 먼저 잡습니다.
current_file_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'

# 만약 현재 폴더 이름이 'notebooks'라면, 한 단계 상위 폴더를 루트로 잡고, 아니라면 현재 폴더를 루트로 잡습니다.
if os.path.basename(current_file_dir) == 'notebooks':
    project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
else:
    project_root_dir = os.path.abspath(current_file_dir)

# 파이썬 최상위 탐색 경로에 프로젝트 루트 리스트를 가장 먼저(0번 인덱스) 찔러 넣습니다.
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

# 이제 안정적으로 src 패키지와 팀원의 evaluation 모듈을 불러옵니다.
from src.preprocessing.loader import extract_pdf
from src.evaluation.retrieval import (
    make_default_ground_truth_dataframe, 
    evaluate_retrieval_dataframe, 
    summarize_retrieval
)

load_dotenv()

# =========================================================
# 2. 예원님의 마크다운 기반 의미 단위 청킹 및 벡터 DB 구축
# =========================================================
config_path = os.path.join(project_root_dir, "config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

yaml_file_dir = config['path']['file_dir'].replace("../", "")
filepath = os.path.join(project_root_dir, yaml_file_dir, "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf")
md_text = extract_pdf(filepath, pages=None)

headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
semantic_chunks = header_splitter.split_text(md_text)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, 
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

final_documents = []
chunk_idx = 0
for chunk in semantic_chunks:
    splitted_texts = text_splitter.split_text(chunk.page_content)
    for text in splitted_texts:
        meta = {
            "source": "고려대학교_RFP_의미단위_B",
            "chunk_id": int(chunk_idx),
            "h1": chunk.metadata.get("Header_1", "표지/개요"),
            "h2": chunk.metadata.get("Header_2", ""),
            "h3": chunk.metadata.get("Header_3", "")
        }
        final_documents.append(Document(page_content=text, metadata=meta))
        chunk_idx += 1

print(f"📊 [의미단위-B] 총 생성된 청크 수: {len(final_documents)}개")

print("📥 OpenAI Embedding API 연결 및 벡터 DB 재구축 중...")
embeddings_b = OpenAIEmbeddings(model="text-embedding-3-small")
persist_db_semantic_b = os.path.join(project_root_dir, "chroma_db_semantic_b")

if os.path.exists(persist_db_semantic_b):
    import shutil
    shutil.rmtree(persist_db_semantic_b)

vector_db = Chroma.from_documents(
    documents=final_documents,
    embedding=embeddings_b,
    persist_directory=persist_db_semantic_b
)
print("✅ 벡터 DB 재구축 완료!")


# =========================================================
# 3. 팀원 평가 표준 템플릿(10개 문항) 기반 실시간 검색 및 본문 매핑
# =========================================================
print("\n🧪 [통합 평가] 팀원 표준 템플릿 기반 실시간 Retrieval 구동...")
eval_df = make_default_ground_truth_dataframe() 

retrieved_results = []
for idx, row in eval_df.iterrows():
    query = row['question']
    retrieved_docs = vector_db.similarity_search(query, k=4)
    
    contexts_list = [doc.page_content for doc in retrieved_docs]
    retrieved_results.append(contexts_list)

eval_df['retrieved_contexts'] = retrieved_results
eval_df['strategy'] = "마크다운 기반 의미 단위 Chunking"


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
print("📊 [의미 단위 청킹 + 팀원 통합 평가 모듈 실시간 정량 지표]")
print("="*80)
print(report_df[['question_id', 'context_recall', 'context_precision', 'mrr', 'ndcg', 'chunk_relevance_scores']].to_markdown(index=False))

print("\n🏆 [전체 평균 요약 스코어]")
print(summary_df.to_markdown(index=False))