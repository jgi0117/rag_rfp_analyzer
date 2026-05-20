import sys
import os
import time
import yaml
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

current_file_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
sys.path.append(project_root_dir)

from src.preprocessing.loader import extract_pdf
from src.preprocessing.cleaner import RFPTextCleaner

# 1. 설정 로드
config_path = os.path.join(project_root_dir, "config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

config['preprocessing']['chunk_size'] = 500
config['preprocessing']['chunk_overlap'] = 0

# 2. 데이터 청킹 (중요: 이 흐름이 주석 처리 없이 온전히 가동되어야 합니다)
yaml_file_dir = config['path']['file_dir'].replace("../", "")
filepath = os.path.join(project_root_dir, yaml_file_dir, "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf")
md_text = extract_pdf(filepath, pages=None)

cleaner = RFPTextCleaner(config=config)
pure_python_chunks = cleaner.run_fixed_size_chunking(md_text, project_name="고려대_차세대포털")
print(f"📊 총 생성된 청크 수: {len(pure_python_chunks)}개")

documents = [
    Document(page_content=chunk, metadata={"source": "고려대학교_RFP", "chunk_id": int(i)}) # ID 강제 int 변환
    for i, chunk in enumerate(pure_python_chunks)
]

# 3. 시나리오 A DB 빌드
print("\n📥 [시나리오 A] bge-m3 로컬 임베딩 모델 연결 및 DB 구축 시작...")
embeddings_a = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

persist_db_a = os.path.join(project_root_dir, "chroma_db_scenario_a")

# 안전을 위해 기존에 깨졌을지 모를 오래된 폴더를 완전히 초기화 후 재생성합니다.
if os.path.exists(persist_db_a):
    import shutil
    shutil.rmtree(persist_db_a)
    print("🧹 기존의 불완전한 DB 폴더를 초기화했습니다.")

start_db_a = time.time()
vector_db_a = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_a,
    persist_directory=persist_db_a
)
print(f"✅ [시나리오 A] 벡터 DB 구축 및 물리 저장 완료! (소요 시간: {time.time() - start_db_a:.2f}초)")

# 4. 즉각적인 임시 검색 테스트 디버깅
query = "서울캠퍼스와 세종캠퍼스의 교직원 현황 및 전임교원 수는 어떻게 되나요?"
print(f"\n🔍 [시나리오 A 자체 빌드 테스트] 질문: '{query}'")
retrieved_docs_a = vector_db_a.similarity_search(query, k=2)

print("="*50)
if not retrieved_docs_a:
    print("🚨 경고: 빌드는 성공했으나 검색 문서가 전혀 반환되지 않습니다! 코드를 점검해야 합니다.")
else:
    for idx, doc in enumerate(retrieved_docs_a):
        print(f"🌟 [테스트 청크 {idx+1}] (원본 Chunk ID: {doc.metadata.get('chunk_id')})")
        print(doc.page_content[:100] + "...")
print("="*50)