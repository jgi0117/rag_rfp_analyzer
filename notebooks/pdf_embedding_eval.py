# -*- coding: utf-8 -*-
"""
pdf_embedding_eval.py
======================
청킹 전략별 임베딩 검색 품질 평가 - 벡터 + BM25 하이브리드(RRF)

전략 A(300/50) · B(500/100) · C(800/150)을 text-embedding-3-small로 임베딩하여
'질문답변.csv'의 실제 QA 쌍으로 Recall · Precision · MRR · nDCG를 비교합니다.
순수 벡터 검색과 BM25+벡터 RRF 하이브리드 검색 결과를 나란히 출력합니다.

실행:
    python pdf_embedding_eval.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pymupdf4llm
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

# ── 설정 ─────────────────────────────────────────────────────────────────────

PDF_PATH = Path(__file__).parent / "data" / "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"
QA_PATH  = next(Path(__file__).parent.glob("*질문답변*"), None)
OUT_DIR  = Path(__file__).parent / "output" / "embedding_eval"

STRATEGIES = [
    {"name": "A_300_50",  "label": "A-소형(300/50)",  "chunk_size": 300, "overlap":  50},
    {"name": "B_500_100", "label": "B-기본(500/100)", "chunk_size": 500, "overlap": 100},
    {"name": "C_800_150", "label": "C-대형(800/150)", "chunk_size": 800, "overlap": 150},
]

EMBED_MODEL      = "text-embedding-3-small"
EMBED_DIM        = 1536
TOP_K            = [1, 3, 5, 10]
BATCH_SIZE       = 100
RELEVANCE_THRESH = 0.45   # ground_truth 5-char n-gram의 몇 % 이상이 청크에 있으면 관련 있음
MAX_EMBED_CHARS  = 6000   # text-embedding-3-small 8192 토큰 한도 대응 (한글 ~3자/토큰)
RRF_K            = 60     # RRF 표준 상수


# ── QA 데이터 로드 ────────────────────────────────────────────────────────────

def load_qa(path: Path) -> list[dict]:
    """번호, 질문(question), 정확한 정답(ground_truth) 컬럼 CSV 로드."""
    qa_pairs = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("질문(question)", "").strip()
            a = row.get("정확한 정답(ground_truth)", "").strip()
            if q and a:
                qa_pairs.append({"question": q, "ground_truth": a})
    return qa_pairs


# ── 전처리 ─────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    # "제 안 요 청 서"처럼 단일 한글 글자가 공백으로 분리된 경우만 병합
    # 일반 어절("총 사업예산은") 사이 공백은 유지
    text = re.sub(
        r"(?<![가-힣])([가-힣])( [가-힣])+(?![가-힣])",
        lambda m: m.group().replace(" ", ""),
        text,
    )
    text = text.replace("**==> picture intentionally omitted <==**", "")
    text = re.sub(r" {3,}", "  ", text)
    return text.strip()


# ── 표 처리 ───────────────────────────────────────────────────────────────────

_TABLE_BLOCK_RE = re.compile(r'(?:[ \t]*\|[^\n]+\n){2,}', re.MULTILINE)


def flatten_table(block: str) -> str:
    """마크다운 표 블록을 평문으로 변환.

    헤더가 있는 표: '헤더: 값' 형식으로 변환.
    헤더가 없거나 빈 셀인 표(요구사항 정의 표 등): 모든 셀 내용을 공백으로 연결.
    """
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    lines = [l for l in lines if not re.match(r'^\|[\s\-|:]+\|$', l)]
    if not lines:
        return ""

    def _cells(line: str) -> list[str]:
        return [c.strip() for c in line.split("|") if c.strip()]

    headers = _cells(lines[0])
    # 헤더가 없거나 1개 이하면 모든 셀 내용을 단순 연결 (요구사항 정의표 등)
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
        pairs = [f"{h}: {v}" for h, v in zip(headers, cells) if v]
        extras = cells[len(headers):]
        row_text = (", ".join(pairs) + (" " + " ".join(extras) if extras else "")).strip()
        if row_text:
            rows.append(row_text)
    return ". ".join(rows)


# ── 슬라이딩 윈도우 청킹 ──────────────────────────────────────────────────────

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


def build_chunks(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """
    산문은 슬라이딩 윈도우로, 마크다운 표는 평문 변환 후 개별 청크로 추가.
    표 블록을 산문에서 제거한 뒤 청킹하므로 표 셀이 산문 청크에 잘려 섞이지 않습니다.
    """
    tables: list[str] = []

    def _extract(m: re.Match) -> str:
        flat = flatten_table(m.group())
        if flat.strip():
            tables.append(flat)
        return "\n"

    prose = _TABLE_BLOCK_RE.sub(_extract, text)
    prose_chunks = sliding_window(prose, chunk_size, overlap)
    for c in prose_chunks:
        c["type"] = "prose"

    start_idx = len(prose_chunks)
    table_chunks = [
        {"index": start_idx + i, "text": t, "type": "table"}
        for i, t in enumerate(tables)
        if t.strip()
    ]
    return prose_chunks + table_chunks


# ── 임베딩 ────────────────────────────────────────────────────────────────────

def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t[:MAX_EMBED_CHARS] for t in texts[i : i + BATCH_SIZE]]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(r.embedding for r in resp.data)
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.05)
    return vectors


# ── Qdrant 컬렉션 구성 ────────────────────────────────────────────────────────

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


# ── BM25 인덱스 구성 ──────────────────────────────────────────────────────────

def build_bm25(chunks: list[dict]) -> BM25Okapi:
    """어절 단위로 토크나이즈하여 BM25 인덱스 구성."""
    tokenized = [c["text"].split() for c in chunks]
    return BM25Okapi(tokenized)


# ── 관련성 판단 ───────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """공백 제거 (한글 n-gram 비교 전처리)."""
    return re.sub(r"\s+", "", text)


def is_relevant(chunk_text: str, ground_truth: str, threshold: float = RELEVANCE_THRESH) -> bool:
    """
    5-char n-gram overlap으로 관련성 판단.
    공백을 제거하고 비교하므로 전처리 차이에 강건합니다.
    """
    n = 5
    chunk_norm = _normalize(chunk_text)
    gt_norm    = _normalize(ground_truth)

    if len(gt_norm) < n:
        return gt_norm in chunk_norm

    gt_ngrams    = {gt_norm[i : i + n] for i in range(len(gt_norm) - n + 1)}
    chunk_ngrams = {chunk_norm[i : i + n] for i in range(len(chunk_norm) - n + 1)}

    return len(gt_ngrams & chunk_ngrams) / len(gt_ngrams) >= threshold


# ── IR 지표 ───────────────────────────────────────────────────────────────────

def recall_at_k(texts: list[str], gt: str, k: int) -> float:
    return float(any(is_relevant(t, gt) for t in texts[:k]))


def precision_at_k(texts: list[str], gt: str, k: int) -> float:
    hits = sum(1 for t in texts[:k] if is_relevant(t, gt))
    return hits / k


def mrr_score(texts: list[str], gt: str) -> float:
    for rank, t in enumerate(texts, 1):
        if is_relevant(t, gt):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(texts: list[str], gt: str, k: int, n_relevant: int = 1) -> float:
    gains = [1.0 if is_relevant(t, gt) else 0.0 for t in texts[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_k = min(n_relevant, k)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ── RRF 하이브리드 검색 ───────────────────────────────────────────────────────

def hybrid_search(
    qdrant: QdrantClient,
    collection: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    query_vector: list[float],
    query_text: str,
    limit: int,
) -> list[str]:
    """
    BM25 + 벡터 검색 결과를 RRF(Reciprocal Rank Fusion)로 결합.

    RRF 공식: score(d) = sum_r 1 / (RRF_K + rank_r(d))
    두 검색기 각각 상위 limit개 결과를 사용합니다.
    """
    # 벡터 검색
    vec_results = qdrant.query_points(
        collection_name=collection, query=query_vector, limit=limit
    )
    vec_ids = [r.payload["index"] for r in vec_results.points]

    # BM25 검색
    query_tokens = query_text.split()
    bm25_scores  = bm25.get_scores(query_tokens)
    bm25_top_ids = list(np.argsort(bm25_scores)[::-1][:limit])

    # RRF 점수 합산
    rrf_scores: dict[int, float] = {}
    for rank, doc_id in enumerate(vec_ids, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    for rank, doc_id in enumerate(bm25_top_ids, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

    # RRF 점수 기준 정렬
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:limit]

    idx_to_text = {c["index"]: c["text"] for c in chunks}
    return [idx_to_text[i] for i in sorted_ids if i in idx_to_text]


# ── 평가 함수 (공통) ──────────────────────────────────────────────────────────

def _compute_metrics(
    texts: list[str], gt: str, k_values: list[int], chunks: list[dict]
) -> dict:
    """주어진 retrieved texts로 모든 지표 계산. chunks로 관련 청크 총수를 산출해 nDCG 정규화."""
    n_rel = max(sum(1 for c in chunks if is_relevant(c["text"], gt)), 1)
    first_hit = next(
        (rank for rank, t in enumerate(texts, 1) if is_relevant(t, gt)), None
    )
    row: dict = {"first_hit": first_hit, "MRR": mrr_score(texts, gt)}
    for k in k_values:
        row[f"Recall@{k}"]    = recall_at_k(texts, gt, k)
        row[f"Precision@{k}"] = precision_at_k(texts, gt, k)
        row[f"nDCG@{k}"]      = ndcg_at_k(texts, gt, k, n_rel)
    return row


def evaluate(
    qdrant: QdrantClient,
    collection: str,
    chunks: list[dict],
    qa_pairs: list[dict],
    query_vectors: list[list[float]],
    k_values: list[int],
) -> tuple[dict[str, float], list[dict]]:
    """순수 벡터 검색 평가."""
    max_k = max(k_values)
    bucket: dict[str, list[float]] = {}
    per_query: list[dict] = []

    for i, (qa, qv) in enumerate(zip(qa_pairs, query_vectors)):
        results = qdrant.query_points(collection_name=collection, query=qv, limit=max_k)
        texts   = [r.payload["text"] for r in results.points]
        row = _compute_metrics(texts, qa["ground_truth"], k_values, chunks)
        row.update({"q_no": i + 1, "question": qa["question"]})
        per_query.append(row)

        for key, val in row.items():
            if key in ("q_no", "question", "first_hit"):
                continue
            bucket.setdefault(key, []).append(val)

    aggregated = {key: float(np.mean(vals)) for key, vals in bucket.items()}
    return aggregated, per_query


def evaluate_hybrid(
    qdrant: QdrantClient,
    collection: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    qa_pairs: list[dict],
    query_vectors: list[list[float]],
    k_values: list[int],
) -> tuple[dict[str, float], list[dict]]:
    """BM25 + 벡터 RRF 하이브리드 검색 평가."""
    max_k = max(k_values)
    bucket: dict[str, list[float]] = {}
    per_query: list[dict] = []

    for i, (qa, qv) in enumerate(zip(qa_pairs, query_vectors)):
        texts = hybrid_search(
            qdrant, collection, bm25, chunks, qv, qa["question"], max_k
        )
        row = _compute_metrics(texts, qa["ground_truth"], k_values, chunks)
        row.update({"q_no": i + 1, "question": qa["question"]})
        per_query.append(row)

        for key, val in row.items():
            if key in ("q_no", "question", "first_hit"):
                continue
            bucket.setdefault(key, []).append(val)

    aggregated = {key: float(np.mean(vals)) for key, vals in bucket.items()}
    return aggregated, per_query


# ── 출력 헬퍼 ────────────────────────────────────────────────────────────────

def print_hit_table(
    qa_pairs: list[dict],
    results_vec: dict[str, list[dict]],
    results_hyb: dict[str, list[dict]],
) -> None:
    """전략별 · 모드별 first_hit / MRR 테이블 출력."""
    col_w = 16
    q_w   = 22

    # 헤더 행 (전략 이름)
    print(f"\n  {'Q':<4} {'질문(앞 22자)':<{q_w}}", end="")
    for s in STRATEGIES:
        label = s["label"][:10]
        print(f"  {'[벡터]'+label:>{col_w}}", end="")
        print(f"  {'[하이브]'+label:>{col_w}}", end="")
    print()
    sep = f"  {'─'*4} {'─'*q_w}"
    for _ in STRATEGIES:
        sep += f"  {'─'*col_w}  {'─'*col_w}"
    print(sep)

    for i, qa in enumerate(qa_pairs):
        q_short = qa["question"][:q_w]
        print(f"  Q{i+1:02d} {q_short:<{q_w}}", end="")
        for s in STRATEGIES:
            for per_q in [results_vec[s["name"]], results_hyb[s["name"]]]:
                row  = per_q[i]
                hit  = row["first_hit"]
                mrr  = row["MRR"]
                cell = f"hit@{hit}/MRR={mrr:.2f}" if hit else "miss /MRR=0.00"
                print(f"  {cell:>{col_w}}", end="")
        print()


def print_recall_table(
    qa_pairs: list[dict],
    results_vec: dict[str, list[dict]],
    results_hyb: dict[str, list[dict]],
    k: int,
) -> None:
    mkey  = f"Recall@{k}"
    col_w = 16
    q_w   = 22

    print(f"\n  [{mkey}]")
    print(f"  {'Q':<4} {'질문(앞 22자)':<{q_w}}", end="")
    for s in STRATEGIES:
        label = s["label"][:8]
        print(f"  {'[V]'+label:>{col_w}}", end="")
        print(f"  {'[H]'+label:>{col_w}}", end="")
    print()

    for i, qa in enumerate(qa_pairs):
        q_short = qa["question"][:q_w]
        print(f"  Q{i+1:02d} {q_short:<{q_w}}", end="")
        for s in STRATEGIES:
            for per_q in [results_vec[s["name"]], results_hyb[s["name"]]]:
                v    = per_q[i][mkey]
                mark = "O" if v > 0 else "-"
                print(f"  {mark:>{col_w}}", end="")
        print()


def print_aggregated_table(
    agg_vec: dict[str, dict],
    agg_hyb: dict[str, dict],
) -> None:
    metric_order = (
        [f"Recall@{k}"    for k in TOP_K] +
        [f"Precision@{k}" for k in TOP_K] +
        [f"nDCG@{k}"      for k in TOP_K] +
        ["MRR"]
    )
    col_w = 11

    print(f"\n{'='*80}")
    print("  전략별 집계 (평균)  [V]=벡터, [H]=하이브리드, *=최고값")
    print(f"{'='*80}")

    # 헤더
    print(f"\n  {'지표':<16}", end="")
    for s in STRATEGIES:
        label = s["label"][:7]
        print(f"  {'[V]'+label:>{col_w}}", end="")
        print(f"  {'[H]'+label:>{col_w}}", end="")
    print()
    print(f"  {'─'*16}", end="")
    for _ in STRATEGIES:
        print(f"  {'─'*col_w}  {'─'*col_w}", end="")
    print()

    for mkey in metric_order:
        all_vals = (
            [agg_vec[s["name"]].get(mkey, 0.0) for s in STRATEGIES] +
            [agg_hyb[s["name"]].get(mkey, 0.0) for s in STRATEGIES]
        )
        best = max(all_vals)
        print(f"  {mkey:<16}", end="")
        for s in STRATEGIES:
            for agg in [agg_vec, agg_hyb]:
                v      = agg[s["name"]].get(mkey, 0.0)
                marker = " *" if v == best else "  "
                print(f"  {v:>{col_w-2}.4f}{marker}", end="")
        print()


# ── 표 청크 덤프 ──────────────────────────────────────────────────────────────

def dump_table_chunks(chunks: list[dict], qa_pairs: list[dict], out_path: Path) -> None:
    """표 청크 전체를 텍스트 파일로 저장. 각 청크마다 QA별 n-gram overlap도 함께 출력."""
    n = 5
    table_chunks = [c for c in chunks if c.get("type") == "table"]

    def _ngram_overlap(chunk_text: str, gt: str) -> float:
        cn = _normalize(chunk_text)
        gn = _normalize(gt)
        if len(gn) < n:
            return float(gn in cn)
        gt_ng    = {gn[i:i+n] for i in range(len(gn)-n+1)}
        chunk_ng = {cn[i:i+n] for i in range(len(cn)-n+1)}
        return len(gt_ng & chunk_ng) / len(gt_ng)

    lines = [
        f"표 청크 덤프  (총 {len(table_chunks)}개)\n",
        f"임계값: {RELEVANCE_THRESH}\n",
        "=" * 72,
    ]
    for c in table_chunks:
        lines.append(f"\n[TABLE CHUNK #{c['index']}]")
        lines.append(c["text"])
        lines.append("")
        overlaps = []
        for i, qa in enumerate(qa_pairs, 1):
            ov = _ngram_overlap(c["text"], qa["ground_truth"])
            mark = " *** 통과" if ov >= RELEVANCE_THRESH else ""
            overlaps.append(f"  Q{i:02d} overlap={ov:.3f}{mark}  gt={qa['ground_truth'][:50]}")
        lines.extend(overlaps)
        lines.append("-" * 72)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    표 청크 덤프 저장: {out_path}  ({len(table_chunks)}개 표 청크)")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # .env 로드
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
    if QA_PATH is None:
        raise FileNotFoundError("질문답변.csv 파일을 찾을 수 없습니다.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    openai_client = OpenAI(api_key=api_key)
    qdrant        = QdrantClient(":memory:")

    print("=" * 72)
    print("  PDF 청킹 전략 x 임베딩 검색 품질 평가  (벡터 + BM25 하이브리드)")
    print(f"  임베딩 모델: {EMBED_MODEL}  |  RRF_K={RRF_K}")
    print("=" * 72)

    # ── 1. QA 데이터 로드 ────────────────────────────────────────────────────
    qa_pairs = load_qa(QA_PATH)
    print(f"\n[1] QA 데이터 로드: {QA_PATH.name}  ->  {len(qa_pairs)}개")
    for i, qa in enumerate(qa_pairs, 1):
        print(f"    Q{i:02d}: {qa['question'][:60]}...")

    # ── 2. PDF 추출 + 전처리 ────────────────────────────────────────────────
    print("\n[2] PyMuPDF4LLM 추출 + 전처리 중...")
    md_text = pymupdf4llm.to_markdown(str(PDF_PATH))
    md_text = preprocess(md_text)
    print(f"    {len(md_text):,}자 / {len(md_text.split()):,}어절")

    # ── 3. 전략별 청킹 + 임베딩 + 컬렉션 + BM25 구성 ────────────────────────
    print("\n[3] 전략별 청킹 · 임베딩 · Qdrant · BM25 구성...")
    strategy_data: dict[str, dict] = {}   # name -> {chunks, bm25}
    for s in STRATEGIES:
        chunks = build_chunks(md_text, s["chunk_size"], s["overlap"])
        print(f"\n  [{s['label']}]  {len(chunks)}청크 (산문+표)  임베딩 중...", end=" ", flush=True)
        vectors = embed_texts(openai_client, [c["text"] for c in chunks])
        build_collection(qdrant, s["name"], chunks, vectors)
        bm25 = build_bm25(chunks)
        strategy_data[s["name"]] = {"chunks": chunks, "bm25": bm25}
        print("완료 (Qdrant + BM25)")

    # ── 3-1. 표 청크 덤프 (B 전략 기준) ────────────────────────────────────
    print("\n[3-1] 표 청크 내용 파일 저장...")
    dump_table_chunks(
        strategy_data["B_500_100"]["chunks"],
        qa_pairs,
        OUT_DIR / "table_chunks_B.txt",
    )

    # ── 4. 쿼리 임베딩 ──────────────────────────────────────────────────────
    print("\n[4] 질문 임베딩 중...")
    query_vectors = embed_texts(openai_client, [qa["question"] for qa in qa_pairs])
    print(f"    {len(query_vectors)}개 완료")

    # ── 5. 전략별 평가 ──────────────────────────────────────────────────────
    print("\n[5] 검색 품질 평가 중...")
    agg_vec:  dict[str, dict]       = {}
    agg_hyb:  dict[str, dict]       = {}
    per_q_vec: dict[str, list[dict]] = {}
    per_q_hyb: dict[str, list[dict]] = {}

    for s in STRATEGIES:
        sd = strategy_data[s["name"]]
        print(f"  {s['label']}: 벡터 ...", end=" ", flush=True)
        av, pv = evaluate(qdrant, s["name"], sd["chunks"], qa_pairs, query_vectors, TOP_K)
        print("완료  |  하이브리드 ...", end=" ", flush=True)
        ah, ph = evaluate_hybrid(
            qdrant, s["name"], sd["bm25"], sd["chunks"],
            qa_pairs, query_vectors, TOP_K,
        )
        print("완료")
        agg_vec[s["name"]]   = av
        agg_hyb[s["name"]]   = ah
        per_q_vec[s["name"]] = pv
        per_q_hyb[s["name"]] = ph

    # ── 6. 질문별 상세 출력 ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  질문별 상세 결과  [V]=벡터  [H]=하이브리드")
    print("  first_hit: 첫 번째 정답 청크 순위, miss=top-10 미포함")
    print(f"{'='*80}")

    print_hit_table(qa_pairs, per_q_vec, per_q_hyb)

    for k in TOP_K:
        print_recall_table(qa_pairs, per_q_vec, per_q_hyb, k)

    # ── 7. 집계 출력 ─────────────────────────────────────────────────────────
    print_aggregated_table(agg_vec, agg_hyb)

    # ── 8. 파일 저장 ────────────────────────────────────────────────────────
    out_json = OUT_DIR / "eval_results.json"
    out_json.write_text(
        json.dumps(
            {
                "embed_model":      EMBED_MODEL,
                "rrf_k":            RRF_K,
                "qa_count":         len(qa_pairs),
                "relevance_thresh": RELEVANCE_THRESH,
                "k_values":         TOP_K,
                "strategies":       [s["label"] for s in STRATEGIES],
                "aggregated": {
                    s["name"]: {
                        "vector":   agg_vec[s["name"]],
                        "hybrid":   agg_hyb[s["name"]],
                    }
                    for s in STRATEGIES
                },
                "per_query": {
                    s["name"]: {
                        "vector":   per_q_vec[s["name"]],
                        "hybrid":   per_q_hyb[s["name"]],
                    }
                    for s in STRATEGIES
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n  쿼리 수  : {len(qa_pairs)}개  (실제 QA: {QA_PATH.name})")
    print(f"  K 설정   : {TOP_K}")
    print(f"  결과 저장: {out_json}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
