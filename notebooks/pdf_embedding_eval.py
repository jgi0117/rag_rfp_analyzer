# -*- coding: utf-8 -*-
"""
pdf_embedding_eval.py
=====================
PDF/Markdown embedding retrieval evaluation.

Embedding and Qdrant indexing are kept in this file, while retrieval metrics
are delegated to src.evaluation.retrieval.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import pymupdf4llm
import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

current_file_dir = Path(__file__).resolve().parent
project_root_dir = current_file_dir.parent
sys.path.append(str(project_root_dir))

from src.evaluation.retrieval import (
    evaluate_retrieval_dataframe,
    make_default_ground_truth_dataframe,
    summarize_retrieval,
)


config_path = project_root_dir / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def resolve_project_path(path_value: str) -> Path:
    """Resolve config paths relative to the project root."""
    path_value = os.path.expanduser(path_value)
    path = Path(path_value)
    if path.is_absolute():
        return path

    parts = path_value.replace("\\", "/").split("/")
    if parts and parts[0] == project_root_dir.name:
        path_value = "/".join(parts[1:])
    return project_root_dir / path_value


default_markdown_path = resolve_project_path(config["path"]["markdown_file"])
markdown_path = resolve_project_path(os.environ.get("RAG_MARKDOWN_PATH", str(default_markdown_path)))
pdf_path = resolve_project_path(config["path"]["raw_pdf_file"])
out_dir = resolve_project_path(config.get("output", {}).get("evaluation_dir", "outputs/evaluation"))

STRATEGIES = [
    {"name": "A_300_50", "label": "A-small(300/50)", "chunk_size": 300, "overlap": 50},
    {"name": "B_500_100", "label": "B-base(500/100)", "chunk_size": 500, "overlap": 100},
    {"name": "C_800_150", "label": "C-large(800/150)", "chunk_size": 800, "overlap": 150},
]

EMBED_MODEL = config.get("embedding", {}).get("model", "text-embedding-3-small")
EMBED_DIM = 1536
TOP_K = int(os.environ.get("RAG_RETRIEVAL_K", str(config.get("retrieval", {}).get("top_k", 3))))
BATCH_SIZE = 100
MAX_EMBED_CHARS = 6000


def load_env_file() -> None:
    env_file = project_root_dir / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def load_source_text() -> str:
    if markdown_path.exists():
        print(f"Markdown loaded: {markdown_path}")
        return markdown_path.read_text(encoding="utf-8-sig")

    print(f"Markdown not found. Extracting PDF: {pdf_path}")
    return pymupdf4llm.to_markdown(str(pdf_path))


def sliding_window(text: str, chunk_size: int, overlap: int) -> list[dict]:
    words = text.split()
    chunks, start, idx = [], 0, 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append({"index": idx, "text": " ".join(words[start:end])})
        if end == len(words):
            break
        start = end - overlap
        idx += 1
    return chunks


_TABLE_BLOCK_RE = re.compile(r'(?:[ \t]*\|[^\n]+\n){2,}', re.MULTILINE)


def flatten_table(block: str) -> str:
    """Convert a Markdown table block into searchable plain text."""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    lines = [line for line in lines if not re.match(r'^\|[\s\-|:]+\|$', line)]
    if not lines:
        return ""

    def _cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.split("|") if cell.strip()]

    headers = _cells(lines[0])
    if len(headers) <= 1:
        all_cells: list[str] = []
        for line in lines:
            all_cells.extend(_cells(line))
        return " ".join(all_cells)

    rows: list[str] = []
    for data_line in lines[1:]:
        cells = _cells(data_line)
        if not cells:
            continue
        pairs = [f"{header}: {value}" for header, value in zip(headers, cells) if value]
        extras = cells[len(headers):]
        row_text = (", ".join(pairs) + (" " + " ".join(extras) if extras else "")).strip()
        if row_text:
            rows.append(row_text)
    return ". ".join(rows)


def build_chunks(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """
    Keep the original chunking strategy: sliding-window prose chunks plus
    separately flattened Markdown table chunks.
    """
    tables: list[str] = []

    def _extract(match: re.Match) -> str:
        flat = flatten_table(match.group())
        if flat.strip():
            tables.append(flat)
        return "\n"

    prose = _TABLE_BLOCK_RE.sub(_extract, text)
    prose_chunks = sliding_window(prose, chunk_size, overlap)
    for chunk in prose_chunks:
        chunk["type"] = "prose"

    start_idx = len(prose_chunks)
    table_chunks = [
        {"index": start_idx + i, "text": table, "type": "table"}
        for i, table in enumerate(tables)
        if table.strip()
    ]
    return prose_chunks + table_chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t[:MAX_EMBED_CHARS] for t in texts[i : i + BATCH_SIZE]]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(r.embedding for r in resp.data)
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.05)
    return vectors


def build_collection(
    qdrant: QdrantClient,
    name: str,
    chunks: list[dict],
    vectors: list[list[float]],
) -> None:
    if qdrant.collection_exists(name):
        qdrant.delete_collection(name)
    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    qdrant.upsert(
        collection_name=name,
        points=[
            PointStruct(
                id=c["index"],
                vector=v,
                payload={"text": c["text"], "index": c["index"]},
            )
            for c, v in zip(chunks, vectors)
        ],
    )


def build_retrieval_dataframe(
    qdrant: QdrantClient,
    collection: str,
    strategy_name: str,
    qa_df: pd.DataFrame,
    query_vectors: list[list[float]],
) -> pd.DataFrame:
    rows = []
    for (_, qa), query_vector in zip(qa_df.iterrows(), query_vectors):
        results = qdrant.query_points(
            collection_name=collection,
            query=query_vector,
            limit=TOP_K,
        )
        rows.append(
            {
                "strategy": strategy_name,
                "question_id": qa["question_id"],
                "question": qa["question"],
                "ground_truth": qa["ground_truth"],
                "retrieved_contexts": [point.payload["text"] for point in results.points],
                "retrieved_chunk_ids": ", ".join(str(point.payload["index"]) for point in results.points),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    load_env_file()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    out_dir.mkdir(parents=True, exist_ok=True)
    openai_client = OpenAI(api_key=api_key)
    qdrant = QdrantClient(":memory:")

    source_text = load_source_text()
    qa_df = make_default_ground_truth_dataframe()
    query_vectors = embed_texts(openai_client, qa_df["question"].tolist())

    evaluated_frames = []
    summary_frames = []

    for strategy in STRATEGIES:
        chunks = build_chunks(source_text, strategy["chunk_size"], strategy["overlap"])
        print(f"[{strategy['label']}] embedding {len(chunks)} chunks...")
        vectors = embed_texts(openai_client, [chunk["text"] for chunk in chunks])
        build_collection(qdrant, strategy["name"], chunks, vectors)

        retrieval_df = build_retrieval_dataframe(
            qdrant=qdrant,
            collection=strategy["name"],
            strategy_name=strategy["name"],
            qa_df=qa_df,
            query_vectors=query_vectors,
        )
        evaluated_df = evaluate_retrieval_dataframe(retrieval_df)
        summary_df = summarize_retrieval(evaluated_df)
        evaluated_frames.append(evaluated_df)
        summary_frames.append(summary_df)

    all_evaluated_df = pd.concat(evaluated_frames, ignore_index=True)
    all_summary_df = pd.concat(summary_frames, ignore_index=True)

    result_path = out_dir / "pdf_embedding_retrieval_eval_results.csv"
    summary_path = out_dir / "pdf_embedding_retrieval_eval_summary.csv"

    all_evaluated_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    all_summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nRetrieval evaluation complete.")
    print(f"Results: {result_path}")
    print(f"Summary: {summary_path}")
    print("\n[Retrieval Summary]")
    print(all_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
