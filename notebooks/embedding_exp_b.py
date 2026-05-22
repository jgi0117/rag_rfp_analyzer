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
current_file_dir = os.path.dirname(os.path.abspath(__file__)) # notebooks/ 폴더 위치
project_root_dir = os.path.abspath(os.path.join(current_file_dir, "..")) # project_root/ 폴더 위치
sys.path.append(project_root_dir)

from src.preprocessing.loader import extract_pdf
from src.preprocessing.cleaner import RFPTextCleaner


# =========================================================
# 2. config.yaml 설정 파일 로드 및 실험 조건 적용
# =========================================================
config_path = os.path.join(project_root_dir, "config.yaml")

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 실험 조건 (500자, 오버랩 0) 상에서 한 번 더 확실하게 고정
config['preprocessing']['chunk_size'] = 500
config['preprocessing']['chunk_overlap'] = 0

print(f"⚙️ config.yaml 로드 완료! (실험 조건 -> Chunk Size: {config['preprocessing']['chunk_size']})")


# =========================================================
# 3. 데이터 로드 및 고정 크기 청킹 가동
# =========================================================
yaml_file_dir = config['path']['file_dir'].replace("../", "") # "data/files" 형식으로 정제
filepath = os.path.join(project_root_dir, yaml_file_dir, "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf")

# 마크다운 텍스트 추출 및 고정 크기 청킹 실행
md_text = extract_pdf(filepath, pages=None)

cleaner = RFPTextCleaner(config=config)
pure_python_chunks = cleaner.run_fixed_size_chunking(md_text, project_name="고려대_차세대포털")
print(f"📊 총 생성된 청크 수: {len(pure_python_chunks)}개")


# =========================================================
# 4. 랭체인 Document 객체 형식으로 래핑
# =========================================================
# 🌟 변수명을 하단 코드와 일치시키기 위해 documents -> final_documents 로 수정합니다.
final_documents = [
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

# Chroma DB 저장 경로도 절대 경로로 지정
persist_db_b = os.path.join(project_root_dir, "chroma_db_scenario_b")

start_db = time.time()
vector_db_b = Chroma.from_documents(
    documents=final_documents,  # 🌟 변수명 매칭 완료
    embedding=embeddings_b,
    persist_directory=persist_db_b
)
print(f"✅ [시나리오 B] 벡터 DB 구축 및 저장 완료! (소요 시간: {time.time() - start_db:.2f}초)")


print("\n" + "="*70)
print("🔍 [고정 단위 청킹 결과 실전 본문 출력] (실제 제안서 본문 구간)")
print("="*70)

# 본문이 본격적으로 시작되는 5번 청크부터 출력
start_idx = 5
sample_count = 3

for i in range(start_idx, min(start_idx + sample_count, len(final_documents))):
    doc = final_documents[i]
    
    print(f"📂 [Chunk Sample {i+1} / {len(final_documents)}] (Index: {i})")
    print(f"🔹 메타데이터 (Metadata):")
    print(f"  - chunk_id: {doc.metadata['chunk_id']}")
    
    # 🌟 고정 청킹에서는 계층 구조 메타데이터(h1, h2, h3)가 없을 수 있으므로 .get() 안전장치 추가
    h1 = doc.metadata.get('h1', 'N/A')
    h2 = doc.metadata.get('h2', 'N/A')
    h3 = doc.metadata.get('h3', 'N/A')
    print(f"  - 계층 구조: {h1} > {h2} > {h3}")
    
    print(f"  - 출처: {doc.metadata['source']}")
    print(f"🔹 청크 전체 길이: {len(doc.page_content)}자")
    print("-" * 50)
    print("🔹 청크 실제 내용 (Content) [전체 출력]:")
    
    print(doc.page_content.strip())
    print("="*70)