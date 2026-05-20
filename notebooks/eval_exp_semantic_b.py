import os
import sys
import pandas as pd
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
current_file_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
sys.path.append(project_root_dir)

print("🚀 [시나리오-의미단위-B] OpenAI text-embedding-3-small 연결 중...")
embeddings_semantic_b = OpenAIEmbeddings(model="text-embedding-3-small")

persist_db_semantic_b = os.path.join(project_root_dir, "chroma_db_semantic_b")
vector_db = Chroma(persist_directory=persist_db_semantic_b, embedding_function=embeddings_semantic_b)

csv_path = os.path.join(project_root_dir, "data", "eval_qa.csv")
df_eval = pd.read_csv(csv_path)
ground_truth_chunks = [
    [0],         # 문항 1
    [288],       # 문항 2 (예산 계획 관련 청크)
    [205],       # 문항 3 (PER-005 동시사용자 표) 
    [203, 204],  # 문항 4 (PER-003, 004 업무응답시간 관련 청크)
    [54],        # 문항 5 (상세 요구사항 분류기준 표 관련 청크)
    [114],       # 문항 6 (SFR-포털-009 관련 청크)
    [281],       # 문항 7 (기술평가/가격평가 비율 관련 청크)
    [61, 62],    # 문항 8 (STR-001 웹 호환성 관련 청크)
    [120, 121],  # 문항 9 (SFR-학사-013 수강소감관리 표) 
    [91]         # 문항 10 (SFR-모바일-001 모바일공통 표)
]

def evaluate_retrieval(retrieved_ids, gt_ids, k=4):
    rel = [1 if r_id in gt_ids else 0 for r_id in retrieved_ids[:k]]
    precisions = [sum(rel[:i+1]) / (i+1) for i in range(len(rel)) if rel[i] == 1]
    context_precision = np.mean(precisions) if precisions else 0.0
    actual_hits = sum([1 for g_id in gt_ids if g_id in retrieved_ids[:k]])
    context_recall = actual_hits / len(gt_ids) if gt_ids else 0.0
    mrr = 0.0
    for rank, r in enumerate(rel):
        if r == 1: mrr = 1.0 / (rank + 1); break
    dcg = sum([r / np.log2(idx + 2) for idx, r in enumerate(rel)])
    idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(gt_ids), k))])
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return context_precision, context_recall, mrr, ndcg

results = []
print("🧪 [시나리오-의미단위-B] 검색 정량 평가 시작...")
for idx, row in df_eval.iterrows():
    q_num = row['번호']
    query = row['질문(question)']
    gt = ground_truth_chunks[idx]
    
    retrieved_docs = vector_db.similarity_search(query, k=4)
    retrieved_ids = [int(doc.metadata.get('chunk_id')) for doc in retrieved_docs if doc.metadata.get('chunk_id') is not None]
            
    c_prec, c_rec, mrr, ndcg = evaluate_retrieval(retrieved_ids, gt, k=4)
    results.append({
        "번호": q_num, "질문 요약": query[:22] + "...", "검색된_IDs": retrieved_ids, "정답_IDs": gt,
        "Precision": round(c_prec, 4), "Recall": round(c_rec, 4), "MRR": round(mrr, 4), "nDCG": round(ndcg, 4)
    })

df_report = pd.DataFrame(results)
print("\n📊 [의미 단위 청킹 + OpenAI API 실험 리포트]")
print(df_report.to_markdown(index=False))
print(f"\n🏆 [최종 스코어] P: {df_report['Precision'].mean():.4f} | R: {df_report['Recall'].mean():.4f} | MRR: {df_report['MRR'].mean():.4f} | nDCG: {df_report['nDCG'].mean():.4f}")