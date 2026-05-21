import sys
import os
import time
import yaml
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

from src.preprocessing.loader import extract_pdf
from src.preprocessing.cleaner import RFPTextCleaner


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


# =========================================================
# 3. 데이터 로드 및 고정 크기 청킹 가동
# =========================================================
# config에 적힌 상대 경로를 시스템 절대 경로로 결합하여 에러를 방지합니다.
yaml_file_dir = config['path']['file_dir'].replace("../", "") # "data/files" 형식으로 정제
filepath = os.path.join(project_root_dir, yaml_file_dir, "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf")

# 마크다운 텍스트 추출 및 예원님 전용 고정 크기 청킹 실행
md_text = extract_pdf(filepath, pages=None)

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