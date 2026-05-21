import sys
import os
import time
import yaml
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # 💡 OpenAI 임베딩 컴포넌트
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
current_file_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
sys.path.append(project_root_dir)

from src.preprocessing.loader import extract_pdf
from src.evaluation.retrieval import (
    evaluate_retrieval_dataframe,
    make_default_ground_truth_dataframe,
    summarize_retrieval,
)

# 1. 데이터 로드 및 마크다운 텍스트 추출
config_path = os.path.join(project_root_dir, "config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def resolve_project_path(path_value: str) -> str:
    """Resolve config paths relative to the project root."""
    path_value = os.path.expanduser(path_value)
    if os.path.isabs(path_value):
        return path_value

    parts = path_value.replace("\\", "/").split("/")
    if parts and parts[0] == os.path.basename(project_root_dir):
        path_value = "/".join(parts[1:])
    return os.path.join(project_root_dir, path_value)

default_markdown_path = resolve_project_path(config["path"]["markdown_file"])
markdown_path = resolve_project_path(os.environ.get("RAG_MARKDOWN_PATH", default_markdown_path))

if os.path.exists(markdown_path):
    print(f"📄 저장된 Markdown 로드: {markdown_path}")
    with open(markdown_path, "r", encoding="utf-8-sig") as f:
        md_text = f.read()
else:
    print(f"⚠️ Markdown 파일을 찾지 못해 PDF에서 다시 추출합니다: {markdown_path}")
    filepath = resolve_project_path(config["path"]["raw_pdf_file"])
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
            "source": "고려대학교_RFP_의미단위_B",
            "chunk_id": int(chunk_idx),
            "h1": chunk.metadata.get("Header_1", ""),
            "h2": chunk.metadata.get("Header_2", ""),
            "h3": chunk.metadata.get("Header_3", "")
        }
        final_documents.append(Document(page_content=text, metadata=meta))
        chunk_idx += 1

print(f"📊 [시나리오-의미단위-B] 최종 생성된 총 청크 수: {len(final_documents)}개")

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
print(f"✅ [시나리오-의미단위-B] 벡터 DB 구축 완료! (소요 시간: {time.time() - start_time:.2f}초)")

# 4. retrieval 평가 코드 연결
retrieval_k = int(os.environ.get("RAG_RETRIEVAL_K", str(config.get("retrieval", {}).get("top_k", 3))))
strategy_name = "scenario_semantic_b_openai_header_1000_overlap_100"

ground_truth_df = make_default_ground_truth_dataframe()
retrieval_rows = []

for _, row in ground_truth_df.iterrows():
    question = row["question"]
    retrieved_docs = vector_db_semantic_b.similarity_search(question, k=retrieval_k)
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

output_dir = os.path.join(project_root_dir, "outputs", "evaluation")
os.makedirs(output_dir, exist_ok=True)

retrieval_output_path = os.path.join(output_dir, "scenario_semantic_b_retrieval_eval_results.csv")
summary_output_path = os.path.join(output_dir, "scenario_semantic_b_retrieval_eval_summary.csv")

evaluated_df.to_csv(retrieval_output_path, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_output_path, index=False, encoding="utf-8-sig")

print("\n✅ [시나리오-의미단위-B] retrieval 평가 완료!")
print(f"📊 평가 결과 저장: {retrieval_output_path}")
print(f"📈 요약 결과 저장: {summary_output_path}")
print("\n[retrieval 평가 요약]")
print(summary_df.to_string(index=False))
