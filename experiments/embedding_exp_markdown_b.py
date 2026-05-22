import argparse
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
    return os.path.join(project_root_dir, path_value)


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str) -> dict:
    config_path = resolve_project_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        experiment_config = yaml.safe_load(f)

    base_config_path = experiment_config.get("base_config")
    if not base_config_path:
        return experiment_config

    with open(resolve_project_path(base_config_path), "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    return deep_merge(base_config, experiment_config)


parser = argparse.ArgumentParser(description="Scenario B markdown embedding experiment")
parser.add_argument("--config", default="configs/experiments/markdown_b.yaml")
args = parser.parse_args()

config = load_config(args.config)
print(
    "config loaded "
    f"(splitter: {config['preprocessing']['splitter']}, "
    f"chunk_size: {config['preprocessing']['chunk_size']}, "
    f"overlap: {config['preprocessing']['chunk_overlap']})"
)

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
chunks = cleaner.run_markdown_chunking(md_text, project_name=project_name)
print(f"Total chunks: {len(chunks)}")

documents = [
    Document(page_content=chunk, metadata={"source": project_name, "chunk_id": i})
    for i, chunk in enumerate(chunks)
]

chunk_output_path = resolve_project_path(config["output"]["chunk_results"])
os.makedirs(os.path.dirname(chunk_output_path), exist_ok=True)
chunk_df = pd.DataFrame(
    [
        {
            "chunk_id": doc.metadata["chunk_id"],
            "source": doc.metadata["source"],
            "source_file": pdf_path,
            "chunking_strategy": config["preprocessing"]["splitter"],
            "chunk_length": len(doc.page_content),
            "chunk_text": doc.page_content,
        }
        for doc in documents
    ]
)
chunk_df.to_csv(chunk_output_path, index=False, encoding="utf-8-sig")
print(f"Chunking result saved: {chunk_output_path}")

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

embeddings_b = OpenAIEmbeddings(model=config["embedding"]["model"])
persist_db_b = resolve_project_path(config["retrieval"]["persist_directory"])

start_db = time.time()
vector_db_b = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_b,
    persist_directory=persist_db_b,
)
print(f"Vector DB built and saved ({time.time() - start_db:.2f}s): {persist_db_b}")

query = config["retrieval"]["sample_query"]
print(f"\n[Scenario B test] Question: '{query}'")
retrieved_docs_b = vector_db_b.similarity_search(query, k=2)

print("=" * 50)
for idx, doc in enumerate(retrieved_docs_b):
    print(f"[Retrieved chunk {idx + 1}] (Chunk ID: {doc.metadata['chunk_id']})")
    print(doc.page_content)
    print("-" * 50)

retrieval_k = int(os.environ.get("RAG_RETRIEVAL_K", str(config["retrieval"].get("top_k", 3))))
strategy_name = config["output"]["strategy_name"]

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
            "retrieved_ranked_chunks": json.dumps(retrieved_ranked_chunks, ensure_ascii=False),
        }
    )

retrieval_df = pd.DataFrame(retrieval_rows)
evaluated_df = evaluate_retrieval_dataframe(retrieval_df)
summary_df = summarize_retrieval(evaluated_df)

retrieval_output_path = resolve_project_path(config["output"]["retrieval_eval_results"])
summary_output_path = resolve_project_path(config["output"]["retrieval_eval_summary"])
os.makedirs(os.path.dirname(retrieval_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

evaluated_df.to_csv(retrieval_output_path, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_output_path, index=False, encoding="utf-8-sig")

print("\n[Scenario B Markdown] retrieval evaluation complete")
print(f"Evaluation saved: {retrieval_output_path}")
print(f"Summary saved: {summary_output_path}")
print("\n[Retrieval evaluation summary]")
print(summary_df.to_string(index=False))
