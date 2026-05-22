import sys
import os
import time
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
current_file_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
sys.path.append(project_root_dir)

from src.preprocessing.loader import extract_pdf

# 1. 데이터 로드 및 마크다운 텍스트 추출
config_path = os.path.join(project_root_dir, "config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

yaml_file_dir = config['path']['file_dir'].replace("../", "")
filepath = os.path.join(project_root_dir, yaml_file_dir, "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf")
md_text = extract_pdf(filepath, pages=None)

# 2. 제안서 포맷 맞춤형 계층 구조(Header) 분할
# 헤더가 없는 문서 극초반부 유실을 방지하기 위해 텍스트 전처리 검증 후 분할
headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
semantic_chunks = header_splitter.split_text(md_text)

# 💡 고정 크기 실험과 정당한 비교를 위해 chunk_size를 500으로 최적화 및 맞춤 조정!
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
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

print(f"📊 [시나리오-의미단위-B] 최적화 후 최종 생성된 총 청크 수: {len(final_documents)}개")

# 3. OpenAI text-embedding-3-small 벡터 DB 구축
print("\n📥 [시나리오-의미단위-B] OpenAI Embedding API 로드 중...")
embeddings_b = OpenAIEmbeddings(model="text-embedding-3-small")

persist_db_semantic_b = os.path.join(project_root_dir, "chroma_db_semantic_b")
if os.path.exists(persist_db_semantic_b):
    import shutil
    shutil.rmtree(persist_db_semantic_b)

start_time = time.time()
vector_db_semantic_b = Chroma.from_documents(
    documents=final_documents,
    embedding=embeddings_b,
    persist_directory=persist_db_semantic_b
)
print(f"✅ [시나리오-의미단위-B] 벡터 DB 재구축 완료! (소요 시간: {time.time() - start_time:.2f}초)")

# =========================================================
# 4. [수정 완료] 본문 위주의 청킹 결과 전체 출력하기
# =========================================================
print("\n" + "="*70)
print("🔍 [의미 단위 청킹 결과 실전 본문 출력] (실제 제안서 본문 구간)")
print("="*70)

# 표지나 로고가 있는 극초반(0~3번)을 건너뛰고, 본문이 본격적으로 시작되는 5번 청크부터 출력
start_idx = 5
sample_count = 3

for i in range(start_idx, min(start_idx + sample_count, len(final_documents))):
    doc = final_documents[i]
    
    print(f"📂 [Chunk Sample {i+1} / {len(final_documents)}] (Index: {i})")
    print(f"🔹 메타데이터 (Metadata):")
    print(f"  - chunk_id: {doc.metadata['chunk_id']}")
    print(f"  - 계층 구조: {doc.metadata['h1']} > {doc.metadata['h2']} > {doc.metadata['h3']}")
    print(f"  - 출처: {doc.metadata['source']}")
    print(f"🔹 청크 전체 길이: {len(doc.page_content)}자")
    print("-" * 50)
    print("🔹 청크 실제 내용 (Content) [전체 출력]:")
    
    # 글자 수 제한(들여쓰기 생략) 없이 원본 마크다운 구조를 그대로 출력합니다.
    print(doc.page_content.strip())
    
    print("="*70)