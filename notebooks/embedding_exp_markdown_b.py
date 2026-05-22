import json
import os
import sys
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# =========================================================
# 1. Path setup and module imports
# =========================================================
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
sys.path.append(project_root_dir)

from src.evaluation.retrieval import (
    evaluate_retrieval_dataframe,
    make_default_ground_truth_dataframe,
    summarize_retrieval,
)
from src.preprocessing.cleaner import RFPTextCleaner
from src.preprocessing.loader import extract_pdf


def resolve_project_path(path_value: str) -> str:
    path_value = os.path.expanduser(path_value)
    if os.path.isabs(path_value):
        return path_value

    parts = path_value.replace("\\", "/").split("/")
    if parts and parts[0] == os.path.basename(project_root_dir):
        path_value = "/".join(parts[1:])
    return os.path.join(project_root_dir, path_value)


# =========================================================
# 2. Load config and pin experiment parameters
# =========================================================
config_path = os.path.join(project_root_dir, "config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["preprocessing"]["chunk_size"] = 500
config["preprocessing"]["chunk_overlap"] = 0

print(
    "config.yaml loaded "
    f"(Chunk Size: {config['preprocessing']['chunk_size']}, "
    f"Overlap: {config['preprocessing']['chunk_overlap']})"
)


# =========================================================
# 3. Load PDF as Markdown, then run Markdown chunking
# =========================================================
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
markdown_chunks = cleaner.run_markdown_chunking(md_text, project_name=project_name)
print(f"Total markdown chunks: {len(markdown_chunks)}")


# =========================================================
# 4. Save chunks and wrap as LangChain Documents
# =========================================================
documents = [
    Document(page_content=chunk, metadata={"source": project_name, "chunk_id": i})
    for i, chunk in enumerate(markdown_chunks)
]

chunk_output_path = resolve_project_path(
    config["output"].get(
        "markdown_chunk_results",
        "outputs/chunks/scenario_b_markdown_chunks.csv",
    )
)
os.makedirs(os.path.dirname(chunk_output_path), exist_ok=True)

chunk_df = pd.DataFrame(
    [
        {
            "chunk_id": doc.metadata["chunk_id"],
            "source": doc.metadata["source"],
            "source_file": pdf_path,
            "chunking_strategy": "markdown",
            "chunk_length": len(doc.page_content),
            "chunk_text": doc.page_content,
        }
        for doc in documents
    ]
)
chunk_df.to_csv(chunk_output_path, index=False, encoding="utf-8-sig")
print(f"Chunking result saved: {chunk_output_path}")


# =========================================================
# 5. OpenAI embeddings and Chroma DB
# =========================================================
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

print("Connecting OpenAI embedding model...")
embeddings_b = OpenAIEmbeddings(model=config["embedding"]["model"])

persist_db_b = resolve_project_path(
    config["retrieval"].get("markdown_persist_directory", "chroma_db_markdown_scenario_b")
)

start_db = time.time()
vector_db_b = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_b,
    persist_directory=persist_db_b,
)
print(f"[Scenario B Markdown] Vector DB built and saved ({time.time() - start_db:.2f}s)")


# =========================================================
# 6. Simple retrieval smoke test
# =========================================================
query = config["retrieval"].get(
    "sample_query",
    "서울캠퍼스와 세종캠퍼스의 교직원 현황 및 전임교원 수는 어떻게 되나요?",
)
print(f"\n[Scenario B Markdown test] Question: '{query}'")
retrieved_docs_b = vector_db_b.similarity_search(query, k=2)

print("=" * 50)
for idx, doc in enumerate(retrieved_docs_b):
    print(f"[Retrieved chunk {idx + 1}] (Chunk ID: {doc.metadata['chunk_id']})")
    print(doc.page_content)
    print("-" * 50)


# =========================================================
# 7. Retrieval evaluation
# =========================================================
retrieval_k = int(os.environ.get("RAG_RETRIEVAL_K", str(config["retrieval"].get("top_k", 3))))
strategy_name = "scenario_b_openai_markdown_500_overlap_0"

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
            "retrieved_ranked_chunks": json.dumps(
                retrieved_ranked_chunks,
                ensure_ascii=False,
            ),
        }
    )

retrieval_df = pd.DataFrame(retrieval_rows)
evaluated_df = evaluate_retrieval_dataframe(retrieval_df)
summary_df = summarize_retrieval(evaluated_df)

output_dir = resolve_project_path(config["output"]["evaluation_dir"])
os.makedirs(output_dir, exist_ok=True)

retrieval_output_path = resolve_project_path(
    config["output"].get(
        "markdown_retrieval_eval_results",
        os.path.join(output_dir, "scenario_b_markdown_retrieval_eval_results.csv"),
    )
)
summary_output_path = resolve_project_path(
    config["output"].get(
        "markdown_retrieval_eval_summary",
        os.path.join(output_dir, "scenario_b_markdown_retrieval_eval_summary.csv"),
    )
)
os.makedirs(os.path.dirname(retrieval_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

evaluated_df.to_csv(retrieval_output_path, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_output_path, index=False, encoding="utf-8-sig")

print("\n[Scenario B Markdown] retrieval evaluation complete")
print(f"Evaluation saved: {retrieval_output_path}")
print(f"Summary saved: {summary_output_path}")
print("\n[Retrieval evaluation summary]")
print(summary_df.to_string(index=False))
