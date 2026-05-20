import os
import sys
import time
import pandas as pd
import numpy as np
import yaml
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # OpenAI API용
from dotenv import load_dotenv

# =========================================================
# 1. 환경 설정 및 시나리오 B DB 로드
# =========================================================
load_dotenv()

current_file_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
sys.path.append(project_root_dir)

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("🚨 OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")

embeddings_b = OpenAIEmbeddings(model="text-embedding-3-small")
persist_db_b = os.path.join(project_root_dir, "chroma_db_scenario_b")
vector_db_b = Chroma(persist_directory=persist_db_b, embedding_function=embeddings_b)

# =========================================================
# 2. 질문답변 데이터셋 및 확정된 Ground Truth Chunk ID 로드
# =========================================================
csv_path = os.path.join(project_root_dir, "data", "eval_qa.csv")
df_eval = pd.read_csv(csv_path)

ground_truth_chunks = [
    [0],        # 1번 질문: 사업개요 및 기간
    [410],      # 2번 질문: 예산 집행/하도급 비율
    [300],      # 3번 질문: 동시 사용자 수 15,000명 (PER-005)
    [296, 298], # 4번 질문: 응답 시간 3초 이내 (PER-002, 003)
    [58],       # 5번 질문: 총 요구사항 수 (SFR 99개 등)
    [117],      # 6번 질문: 지능형 검색 방식 (SFR-포털-009)
    [406],      # 7번 질문: 기술/가격 평가 배점
    [71],       # 8번 질문: 웹 호환성 및 개발 보안가이드 (STR-001/002)
    [166, 167], # 9번 질문: 수강소감관리 (SFR-학사-013)
    [119, 120]  # 10번 질문: 모바일 서비스 통합 (SFR-모바일-001)
]

# =========================================================
# 3. 정량적 평가 매트릭스 계산 수학 로직 함수
# =========================================================
def evaluate_retrieval(retrieved_ids, gt_ids, k=4):
    rel = [1 if r_id in gt_ids else 0 for r_id in retrieved_ids[:k]]
    
    # 1. Context Precision
    precisions = [sum(rel[:i+1]) / (i+1) for i in range(len(rel)) if rel[i] == 1]
    context_precision = np.mean(precisions) if precisions else 0.0
    
    # 2. Context Recall
    actual_hits = sum([1 for g_id in gt_ids if g_id in retrieved_ids[:k]])
    context_recall = actual_hits / len(gt_ids) if gt_ids else 0.0
    
    # 3. MRR
    mrr = 0.0
    for rank, r in enumerate(rel):
        if r == 1:
            mrr = 1.0 / (rank + 1)
            break
            
    # 4. nDCG
    dcg = sum([r / np.log2(idx + 2) for idx, r in enumerate(rel)])
    idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(gt_ids), k))])
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return context_precision, context_recall, mrr, ndcg

# =========================================================
# 4. 평가 루프 가동 및 연산
# =========================================================
results = []
print("🧪 [시나리오 B] RAG 리트리버 정량적 매트릭스 평가 시작 (Top-4)...")

for idx, row in df_eval.iterrows():
    q_num = row['번호']
    query = row['질문(question)']
    gt = ground_truth_chunks[idx]
    
    retrieved_docs = vector_db_b.similarity_search(query, k=4)
    retrieved_ids = [doc.metadata['chunk_id'] for doc in retrieved_docs]
    
    c_prec, c_rec, mrr, ndcg = evaluate_retrieval(retrieved_ids, gt, k=4)
    
    results.append({
        "번호": q_num,
        "질문 요약": query[:22] + "...",
        "검색된_IDs": retrieved_ids,
        "정답_IDs": gt,
        "Precision": round(c_prec, 4),
        "Recall": round(c_rec, 4),
        "MRR": round(mrr, 4),
        "nDCG": round(ndcg, 4)
    })

# =========================================================
# 5. 결과 리포트 출력 및 저장
# =========================================================
df_report = pd.DataFrame(results)
print("\n📊 [500자 고정 크기 청킹 + OpenAI text-embedding-3-small 실험 리포트]")
print(df_report.to_markdown(index=False))

print("\n🏆 [시나리오 B 평균 최종 결과 스코어]")
print(f"🔹 Mean Context Precision : {df_report['Precision'].mean():.4f}")
print(f"🔹 Mean Context Recall    : {df_report['Recall'].mean():.4f}")
print(f"🔹 Mean MRR               : {df_report['MRR'].mean():.4f}")
print(f"🔹 Mean nDCG              : {df_report['nDCG'].mean():.4f}")

output_csv = os.path.join(project_root_dir, "data", "scenario_b_eval_results.csv")
df_report.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n💾 성적서 파일이 성공적으로 보존되었습니다: '{output_csv}'")