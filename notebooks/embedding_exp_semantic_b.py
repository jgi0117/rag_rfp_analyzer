import os
import sys
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

current_file_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
sys.path.append(project_root_dir)

from src.evaluation.generation import evaluate_generation_dataframe, summarize_by_strategy
from src.evaluation.retrieval import make_default_ground_truth_dataframe
from src.preprocessing.loader import extract_pdf


def resolve_project_path(path_value: str) -> str:
    """Resolve config paths relative to the project root."""
    path_value = os.path.expanduser(path_value)
    if os.path.isabs(path_value):
        return path_value

    parts = path_value.replace("\\", "/").split("/")
    if parts and parts[0] == os.path.basename(project_root_dir):
        path_value = "/".join(parts[1:])
    return os.path.join(project_root_dir, path_value)


config_path = os.path.join(project_root_dir, "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

default_markdown_path = resolve_project_path(config["path"]["markdown_file"])
markdown_path = resolve_project_path(os.environ.get("RAG_MARKDOWN_PATH", default_markdown_path))

if os.path.exists(markdown_path):
    print(f"Loaded parsed markdown: {markdown_path}")
    with open(markdown_path, "r", encoding="utf-8-sig") as f:
        md_text = f.read()
else:
    filepath = resolve_project_path(config["path"]["raw_pdf_file"])
    print(f"Markdown not found. Extracting PDF instead: {filepath}")
    md_text = extract_pdf(filepath, pages=None)


headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
semantic_chunks = header_splitter.split_text(md_text)

semantic_chunk_size = int(os.environ.get("RAG_SEMANTIC_CHUNK_SIZE", "1000"))
semantic_chunk_overlap = int(os.environ.get("RAG_SEMANTIC_CHUNK_OVERLAP", "100"))
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=semantic_chunk_size,
    chunk_overlap=semantic_chunk_overlap,
)

final_documents = []
chunk_idx = 0
for chunk in semantic_chunks:
    splitted_texts = text_splitter.split_text(chunk.page_content)
    for text in splitted_texts:
        meta = {
            "source": f"{config['project']['target_document']}_semantic_b",
            "chunk_id": int(chunk_idx),
            "h1": chunk.metadata.get("Header_1", ""),
            "h2": chunk.metadata.get("Header_2", ""),
            "h3": chunk.metadata.get("Header_3", ""),
        }
        final_documents.append(Document(page_content=text, metadata=meta))
        chunk_idx += 1

print(f"[semantic_b] Created {len(final_documents)} chunks.")


embedding_model_name = os.environ.get(
    "RAG_EMBEDDING_MODEL",
    config.get("embedding", {}).get("model", "text-embedding-3-small"),
)
persist_db_semantic_b = resolve_project_path(
    os.environ.get("RAG_SEMANTIC_PERSIST_DIR", "chroma_db_semantic_b")
)

print(f"[semantic_b] Building Chroma DB with OpenAI embeddings: {embedding_model_name}")
embeddings_b = OpenAIEmbeddings(model=embedding_model_name)

start_db = time.time()
vector_db_semantic_b = Chroma.from_documents(
    documents=final_documents,
    embedding=embeddings_b,
    persist_directory=persist_db_semantic_b,
)
print(f"[semantic_b] Vector DB saved: {persist_db_semantic_b} ({time.time() - start_db:.2f}s)")


generation_model_name = os.environ.get(
    "RAG_GENERATION_MODEL",
    config.get("generation", {}).get("model", "gpt-4o-mini"),
)
retrieval_k = int(os.environ.get("RAG_RETRIEVAL_K", config.get("retrieval", {}).get("top_k", 3)))
strategy_name = (
    f"semantic_b_openai_{semantic_chunk_size}_overlap_{semantic_chunk_overlap}"
)

print(
    f"[semantic_b] Generating answers "
    f"(model={generation_model_name}, retrieval_k={retrieval_k})"
)
llm = ChatOpenAI(model=generation_model_name, temperature=0)


def generate_answer(question: str, retrieved_docs: list[Document]) -> str:
    retrieved_context = "\n\n".join(
        f"[Context {idx + 1}]\n{doc.page_content}"
        for idx, doc in enumerate(retrieved_docs)
    )
    prompt = f"""
You are a Korean RFP question-answering assistant.
Answer in Korean, using only the retrieved context below.
If the context does not contain enough evidence, say that you do not know.

[Question]
{question}

[Retrieved context]
{retrieved_context}

[Answer]
""".strip()
    return llm.invoke(prompt).content.strip()


ground_truth_df = make_default_ground_truth_dataframe()
generation_rows = []

start_eval = time.time()
for _, row in ground_truth_df.iterrows():
    question = row["question"]
    ground_truth = row["ground_truth"]

    print(f"[semantic_b] Retrieving and answering question {row['question_id']}...")
    retrieved_docs = vector_db_semantic_b.similarity_search(question, k=retrieval_k)
    retrieved_context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
    retrieved_chunk_ids = ", ".join(
        str(doc.metadata.get("chunk_id", "")) for doc in retrieved_docs
    )
    generated_answer = generate_answer(question, retrieved_docs)

    generation_rows.append(
        {
            "strategy": strategy_name,
            "question_id": row["question_id"],
            "question": question,
            "ground_truth_answer": ground_truth,
            "retrieved_context": retrieved_context,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "generated_answer": generated_answer,
        }
    )

generation_df = pd.DataFrame(generation_rows)
evaluated_df = evaluate_generation_dataframe(generation_df)
summary_df = summarize_by_strategy(evaluated_df)

output_dir = resolve_project_path(config.get("output", {}).get("evaluation_dir", "outputs/evaluation"))
os.makedirs(output_dir, exist_ok=True)

generation_output_path = os.path.join(output_dir, "semantic_b_generation_results.csv")
evaluation_output_path = os.path.join(output_dir, "semantic_b_generation_eval_results.csv")
summary_output_path = os.path.join(output_dir, "semantic_b_generation_eval_summary.csv")

generation_df.to_csv(generation_output_path, index=False, encoding="utf-8-sig")
evaluated_df.to_csv(evaluation_output_path, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_output_path, index=False, encoding="utf-8-sig")

print("[semantic_b] Generation and evaluation complete.")
print(f"Generation results: {generation_output_path}")
print(f"Evaluation results: {evaluation_output_path}")
print(f"Summary results: {summary_output_path}")
print(f"Evaluation elapsed: {time.time() - start_eval:.2f}s")
print(summary_df.to_string(index=False))
