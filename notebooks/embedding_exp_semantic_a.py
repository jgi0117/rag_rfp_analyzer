import sys
import os
import time
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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

# 2. 제안서 포맷 맞춤형 계층 구조(Header) 분할 + 2차 안전 청킹
headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
semantic_chunks = header_splitter.split_text(md_text)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

final_documents = []
chunk_idx = 0
for chunk in semantic_chunks:
    splitted_texts = text_splitter.split_text(chunk.page_content)
    for text in splitted_texts:
        meta = {
            "source": "고려대학교_RFP_의미단위_A",
            "chunk_id": int(chunk_idx),
            "h1": chunk.metadata.get("Header_1", ""),
            "h2": chunk.metadata.get("Header_2", ""),
            "h3": chunk.metadata.get("Header_3", "")
        }
        final_documents.append(Document(page_content=text, metadata=meta))
        chunk_idx += 1

print(f"📊 [시나리오-의미단위-A] 최종 생성된 총 청크 수: {len(final_documents)}개")

# 3. 로컬 ko-sroberta 벡터 DB 구축
print("\n📥 [시나리오-의미단위-A] ko-sroberta 모델 로드 시작...")
embeddings_a = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

persist_db_semantic_a = os.path.join(project_root_dir, "chroma_db_semantic_a")
if os.path.exists(persist_db_semantic_a):
    import shutil
    shutil.rmtree(persist_db_semantic_a)

start_time = time.time()
vector_db_semantic_a = Chroma.from_documents(
    documents=final_documents,
    embedding=embeddings_a,
    persist_directory=persist_db_semantic_a
)
print(f"✅ [시나리오-의미단위-A] 벡터 DB 구축 완료! (소요 시간: {time.time() - start_time:.2f}초)")