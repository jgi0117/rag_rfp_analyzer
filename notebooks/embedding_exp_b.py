import sys
import os
import time
import yaml
import pandas as pd
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
from src.evaluation.generation import evaluate_generation_dataframe, summarize_by_strategy
from src.evaluation.retrieval import make_default_ground_truth_dataframe


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
default_markdown_path = os.path.join(
    project_root_dir,
    "data",
    "parsed",
    "docling",
    "markdown",
    "고려대학교_차세대 포털·학사 정보시스템 구축사업.md",
)
markdown_path = os.environ.get("RAG_MARKDOWN_PATH", default_markdown_path)

if os.path.exists(markdown_path):
    print(f"📄 저장된 Markdown 로드: {markdown_path}")
    with open(markdown_path, "r", encoding="utf-8-sig") as f:
        md_text = f.read()
else:
    # Markdown이 없을 때만 PDF를 다시 파싱합니다.
    print(f"⚠️ Markdown 파일을 찾지 못해 PDF에서 다시 추출합니다: {markdown_path}")
    yaml_file_dir = config['path']['file_dir'].replace("../", "") # "data/files" 형식으로 정제
    filepath = os.path.join(project_root_dir, yaml_file_dir, "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf")
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
# 6. 시나리오 B RAG 생성 및 평가 결과 저장
# =========================================================

generation_model_name = os.environ.get("RAG_GENERATION_MODEL", "gpt-4o-mini")
retrieval_k = int(os.environ.get("RAG_RETRIEVAL_K", "3"))
strategy_name = "scenario_b_openai_500_overlap_0"

print(f"\n🧠 생성 모델 연결 중... (model={generation_model_name}, retrieval_k={retrieval_k})")
llm = ChatOpenAI(model=generation_model_name, temperature=0)


def generate_answer(question: str, retrieved_docs: list[Document]) -> str:
    retrieved_context = "\n\n".join(
        f"[Context {idx + 1}]\n{doc.page_content}"
        for idx, doc in enumerate(retrieved_docs)
    )
    prompt = f"""
당신은 RFP 문서 기반 질의응답 assistant입니다.
아래 검색된 문맥만 근거로 한국어로 간결하게 답변하세요.
문맥에 없는 내용은 추측하지 말고 모른다고 답변하세요.

[질문]
{question}

[검색된 문맥]
{retrieved_context}

[답변]
""".strip()
    return llm.invoke(prompt).content.strip()


ground_truth_df = make_default_ground_truth_dataframe()
generation_rows = []

start_eval = time.time()
for _, row in ground_truth_df.iterrows():
    question = row["question"]
    ground_truth = row["ground_truth"]

    print(f"\n🔍 질문 {row['question_id']} 검색 및 답변 생성 중...")
    retrieved_docs = vector_db_b.similarity_search(question, k=retrieval_k)
    retrieved_context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
    retrieved_chunk_ids = ", ".join(str(doc.metadata.get("chunk_id", "")) for doc in retrieved_docs)
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

output_dir = os.path.join(project_root_dir, "outputs", "evaluation")
os.makedirs(output_dir, exist_ok=True)

generation_output_path = os.path.join(output_dir, "scenario_b_generation_results.csv")
evaluation_output_path = os.path.join(output_dir, "scenario_b_generation_eval_results.csv")
summary_output_path = os.path.join(output_dir, "scenario_b_generation_eval_summary.csv")

generation_df.to_csv(generation_output_path, index=False, encoding="utf-8-sig")
evaluated_df.to_csv(evaluation_output_path, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_output_path, index=False, encoding="utf-8-sig")

print("\n✅ [시나리오 B] 생성 및 평가 완료!")
print(f"📄 생성 결과 저장: {generation_output_path}")
print(f"📊 평가 결과 저장: {evaluation_output_path}")
print(f"📌 요약 결과 저장: {summary_output_path}")
print(f"⏱️ 평가 파이프라인 소요 시간: {time.time() - start_eval:.2f}초")
print("\n[평가 요약]")
print(summary_df.to_string(index=False))
