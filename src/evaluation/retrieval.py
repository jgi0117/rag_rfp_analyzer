"""
RAG 검색 결과를 평가하기 위한 검색 평가 지표 모음.

평가 지표:
- Context recall: 정답을 만드는 데 필요한 정보가 검색된 청킹 안에
  누락 없이 들어있는지 평가한다.
- Context precision: 검색된 청킹들 중 진짜 답변에 필요한 정보가
  상위에 잘 배치되어 있는지 평가한다.
- MRR: 사용자가 원하는 첫 번째 정답 조각이 얼마나 맨 위에 노출되었는지
  순위 기반으로 평가한다.
- nDCG: 검색된 문서들의 관련도 점수와 순서를 모두 고려해 평가한다.
  가장 관련성이 높은 고품질 청킹이 맨 위에 올수록 높은 점수를 준다.

입력 컬럼 예시:
question_id, question, ground_truth, retrieved_contexts

`retrieved_contexts`는 다음 형식을 지원한다:
- 문자열 리스트
- JSON 또는 Python literal 형태의 리스트 문자열
- "\\n---\\n", "|||", "\\n\\n" 중 하나로 구분된 단일 문자열
- retrieved_context_1, retrieved_context_2, ... 형태의 다중 컬럼
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_GROUND_TRUTH_ROWS = [
    {
        "question_id": 1,
        "question": "고려대학교 차세대 포털·학사 정보시스템 구축 사업의 총 사업예산(V.A.T 포함)과 계약일로부터의 사업 기간은 각각 어떻게 명시되어 있나요?",
        "ground_truth": "총 사업예산은 부가가치세 포함 11,270,000,000원이며 사업 기간은 계약일로부터 24개월 이내입니다.",
    },
    {
        "question_id": 2,
        "question": "본 사업의 예산 집행 계획 중 2025학년도에 지급될 예정인 사업비의 비율은 전체의 몇 퍼센트인가요?",
        "ground_truth": "2025학년도에는 전체 사업예산의 약 40%를 분할 지급할 예정입니다.",
    },
    {
        "question_id": 3,
        "question": "성능요구사항(PER-005)에 명시된 기준에 따라 수강신청 시스템과 같이 특정 시간에 사용자가 폭증하는 시스템의 경우 보장해야 하는 최소 동시 사용자 수는 얼마인가요?",
        "ground_truth": "수강신청 시스템 등 트래픽 폭증 시스템의 경우 동시 사용자 15,000명 이상을 지원해야 합니다.",
    },
    {
        "question_id": 4,
        "question": "성능요구사항인 온라인성 업무응답시간(PER-002)과 웹페이지 디스플레이시간(PER-003)에서 목표로 하는 처리 시간은 각각 요청 후 몇 초 이내인가요?",
        "ground_truth": "모든 질의에 대한 결과 처리 및 웹페이지의 완전한 출력은 사용자가 요청한 시간으로부터 3초 내에 완료되어야 합니다.",
    },
    {
        "question_id": 5,
        "question": "상세 요구사항 분류기준에 따른 본 사업의 총 요구사항 수와 그 중 가장 많은 비중을 차지하는 기능요구사항(SFR)의 개수는 각각 얼마인가요?",
        "ground_truth": "본 사업의 총 요구사항 수는 160개이며 이 중 기능요구사항(SFR)은 99개입니다.",
    },
    {
        "question_id": 6,
        "question": "기능요구사항 ID SFR-포털-009 지능형 검색에서 사용자 의도와 요구에 맞게 도입해야 하는 4가지 핵심 검색 방식은 무엇인가요?",
        "ground_truth": "의미기반 검색, 개인화 검색 / 추천 검색, 유사문장 검색, 다국어 검색의 4가지 방식을 도입해야 합니다.",
    },
    {
        "question_id": 7,
        "question": "제안 평가 방식 중 기술평가와 가격평가의 배점 비율은 각각 어떻게 구성되며 기술평가 항목 중 기술 및 기능 부문에 배정된 점수는 몇 점인가요?",
        "ground_truth": "기술평가 90%와 가격평가 10%로 구성되며 기술평가 항목 중 기술 및 기능 부문에는 30점이 배정되어 있습니다.",
    },
    {
        "question_id": 8,
        "question": "표준요구사항 STR-001에 따라 웹 호환성 및 접근성 준수를 위해 만족해야 하는 보안 가이드의 기준 연도와 최종 제출해야 하는 산출물 보고서의 명칭은 무엇인가요?",
        "ground_truth": "행정안전부의 소프트웨어 개발보안 가이드(2021.11) 기준을 만족해야 하며 웹 접근성 결과 보고서를 제출해야 합니다.",
    },
    {
        "question_id": 9,
        "question": "기능요구사항 SFR-학사-013 수강소감관리에서 강의 개선을 위해 관리자가 작성해야 하는 보고서 명칭과 해당 기능에서 종합적으로 조회해야 하는 4가지 이력 항목은 무엇인가요?",
        "ground_truth": "강의개선보고서(CQI)를 작성해야 하며 CQI보고서 내용, 수강소감설문 사후조치 내용, 강의평가 이력, 역량수준 이력을 종합 조회해야 합니다.",
    },
    {
        "question_id": 10,
        "question": "기능요구사항 SFR-모바일-001에 명시된 지침에 따라 기존 모바일 통합앱인 호잇에서 흡수하여 재구축해야 하는 주요 기능 5가지를 기술하세요.",
        "ground_truth": "검색, 교내연락처, 푸시, 일정, 유니버스가 포함되며 이 외에도 식단, 교통서비스 등을 흡수하여 재구축해야 합니다.",
    },
]

STOPWORDS = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "와",
    "과",
    "및",
    "등",
    "그",
    "중",
    "본",
    "각각",
    "어떻게",
    "무엇인가요",
    "얼마인가요",
    "기술하세요",
    "합니다",
    "입니다",
    "되어",
    "있는",
    "위해",
    "경우",
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_token(token: str) -> str:
    """검색 overlap 평가를 위한 간단한 한국어 조사 정규화."""
    token = token.lower()
    if re.fullmatch(r"[가-힣]+", token) and len(token) > 2:
        for suffix in (
            "으로부터",
            "에서는",
            "에게는",
            "에는",
            "에서",
            "으로",
            "로서",
            "로써",
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "의",
            "와",
            "과",
            "로",
        ):
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                return token[: -len(suffix)]
    return token


def tokenize(text: Any, *, remove_stopwords: bool = True) -> List[str]:
    text = normalize_text(text).lower()
    tokens = [normalize_token(token) for token in re.findall(r"[가-힣]+|[a-zA-Z]+|\d+(?:[.,]\d+)*%?", text)]
    if not remove_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _split_context_string(value: str) -> List[str]:
    if value is None:
        return []
    original = str(value).strip()
    if not original or original.lower() == "nan":
        return []

    for separator in ("\n---\n", "|||", "\n\n"):
        if separator in original:
            return [normalize_text(part) for part in original.split(separator) if normalize_text(part)]
    return [normalize_text(original)]


def parse_retrieved_contexts(
    row: Dict[str, Any],
    contexts_col: str = "retrieved_contexts",
    context_prefixes: Sequence[str] = (
        "retrieved_context_",
        "retrieved_chunk_",
        "context_",
        "chunk_",
    ),
) -> List[str]:
    """일반적인 DataFrame/CSV 구조에서 순위가 있는 검색 청크를 추출한다."""
    value = row.get(contexts_col)
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]

    if value is not None and normalize_text(value):
        raw = normalize_text(value)
        if raw.lower() != "nan":
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                try:
                    parsed = ast.literal_eval(raw)
                except (SyntaxError, ValueError):
                    parsed = None

            if isinstance(parsed, list):
                return [normalize_text(item) for item in parsed if normalize_text(item)]
            return _split_context_string(str(value))

    ranked_columns = []
    for col in row:
        for prefix in context_prefixes:
            if col.startswith(prefix):
                suffix = col.removeprefix(prefix)
                rank = int(suffix) if suffix.isdigit() else 10_000
                ranked_columns.append((rank, col))
                break

    contexts = []
    for _, col in sorted(ranked_columns):
        context = normalize_text(row.get(col, ""))
        if context and context.lower() != "nan":
            contexts.append(context)
    return contexts


def compute_context_recall(ground_truth: str, contexts: Sequence[str]) -> float:
    """전체 검색 청크가 정답 정보 토큰을 얼마나 포함하는지 계산한다."""
    gt_counts = Counter(tokenize(ground_truth))
    if not gt_counts:
        return 0.0

    context_tokens = set(tokenize(" ".join(contexts)))
    covered = sum(count for token, count in gt_counts.items() if token in context_tokens)
    return round(safe_div(covered, sum(gt_counts.values())), 4)


def compute_chunk_relevance(ground_truth: str, context: str) -> float:
    """
    정답과 검색 청크의 토큰 overlap을 기반으로 청크 관련도 점수를 계산한다.

    검색 청크는 답변에 필요한 정보 외의 주변 문맥도 함께 포함할 수 있으므로,
    precision보다 recall에 조금 더 높은 가중치를 둔다.
    """
    gt_tokens = tokenize(ground_truth)
    context_tokens = tokenize(context)
    if not gt_tokens or not context_tokens:
        return 0.0

    gt_counts = Counter(gt_tokens)
    context_counts = Counter(context_tokens)
    overlap = sum((gt_counts & context_counts).values())
    recall = safe_div(overlap, len(gt_tokens))
    precision = safe_div(overlap, len(context_tokens))
    relevance = (0.7 * recall) + (0.3 * precision)
    return round(min(relevance, 1.0), 4)


def compute_context_precision(
    relevance_scores: Sequence[float],
    relevance_threshold: float = 0.2,
) -> float:
    """
    검색 청크의 Average Precision을 계산한다.

    청크 관련도 점수가 `relevance_threshold` 이상이면 답변에 유용한 청크로 보고,
    유용한 청크가 상위에 있을수록 높은 점수를 준다.
    """
    relevant_count = 0
    precision_sum = 0.0

    for rank, score in enumerate(relevance_scores, start=1):
        if score >= relevance_threshold:
            relevant_count += 1
            precision_sum += safe_div(relevant_count, rank)

    return round(safe_div(precision_sum, relevant_count), 4)


def compute_mrr(
    relevance_scores: Sequence[float],
    relevance_threshold: float = 0.2,
) -> float:
    for rank, score in enumerate(relevance_scores, start=1):
        if score >= relevance_threshold:
            return round(1 / rank, 4)
    return 0.0


def _dcg(scores: Sequence[float]) -> float:
    return sum(((2**score) - 1) / math.log2(rank + 1) for rank, score in enumerate(scores, start=1))


def compute_ndcg(relevance_scores: Sequence[float], k: Optional[int] = None) -> float:
    if k is not None:
        relevance_scores = relevance_scores[:k]
    if not relevance_scores:
        return 0.0

    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = _dcg(ideal_scores)
    return round(safe_div(_dcg(relevance_scores), ideal_dcg), 4)


def evaluate_retrieval_row(
    row: Dict[str, Any],
    question_col: str = "question",
    ground_truth_col: str = "ground_truth",
    contexts_col: str = "retrieved_contexts",
    relevance_threshold: float = 0.2,
    ndcg_k: Optional[int] = None,
) -> Dict[str, Any]:
    contexts = parse_retrieved_contexts(row, contexts_col=contexts_col)
    ground_truth = normalize_text(row.get(ground_truth_col, ""))
    question = normalize_text(row.get(question_col, ""))
    relevance_scores = [compute_chunk_relevance(ground_truth, context) for context in contexts]

    first_relevant_rank = 0
    for rank, score in enumerate(relevance_scores, start=1):
        if score >= relevance_threshold:
            first_relevant_rank = rank
            break

    return {
        "context_recall": compute_context_recall(ground_truth, contexts),
        "context_precision": compute_context_precision(relevance_scores, relevance_threshold),
        "mrr": compute_mrr(relevance_scores, relevance_threshold),
        "ndcg": compute_ndcg(relevance_scores, ndcg_k),
        "retrieved_count": len(contexts),
        "first_relevant_rank": first_relevant_rank,
        "chunk_relevance_scores": ", ".join(f"{score:.4f}" for score in relevance_scores),
        "question_for_eval": question,
    }


def evaluate_retrieval_dataframe(
    df,
    question_col: str = "question",
    ground_truth_col: str = "ground_truth",
    contexts_col: str = "retrieved_contexts",
    relevance_threshold: float = 0.2,
    ndcg_k: Optional[int] = None,
):
    import pandas as pd

    scores = []
    for _, row in df.iterrows():
        scores.append(
            evaluate_retrieval_row(
                row.to_dict(),
                question_col=question_col,
                ground_truth_col=ground_truth_col,
                contexts_col=contexts_col,
                relevance_threshold=relevance_threshold,
                ndcg_k=ndcg_k,
            )
        )

    score_df = pd.DataFrame(scores)
    return pd.concat([df.reset_index(drop=True), score_df], axis=1)


def summarize_retrieval(
    evaluated_df,
    group_col: Optional[str] = "strategy",
    metric_cols: Optional[Sequence[str]] = None,
):
    metric_cols = list(
        metric_cols
        or [
            "context_recall",
            "context_precision",
            "mrr",
            "ndcg",
            "retrieved_count",
        ]
    )
    if group_col and group_col in evaluated_df.columns:
        return (
            evaluated_df.groupby(group_col)[metric_cols]
            .mean(numeric_only=True)
            .round(4)
            .reset_index()
        )
    return evaluated_df[metric_cols].mean(numeric_only=True).round(4).to_frame("mean").T


def make_default_ground_truth_dataframe():
    import pandas as pd

    return pd.DataFrame(DEFAULT_GROUND_TRUTH_ROWS)


def _load_input(path: str, sheet: Any = 0):
    import pandas as pd

    if path.lower().endswith(".xlsx"):
        sheet_name = int(sheet) if str(sheet).isdigit() else sheet
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 검색 결과 CSV/XLSX 평가")
    parser.add_argument("--input", required=True, help="검색 결과 파일 경로(csv/xlsx)")
    parser.add_argument("--output", default="retrieval_eval_result.csv", help="평가 결과 저장 경로")
    parser.add_argument("--summary", default="retrieval_eval_summary.csv", help="요약 저장 경로")
    parser.add_argument("--sheet", default=0, help="xlsx 입력 시 sheet 이름 또는 index")
    parser.add_argument("--question-col", default="question")
    parser.add_argument("--ground-truth-col", default="ground_truth")
    parser.add_argument("--contexts-col", default="retrieved_contexts")
    parser.add_argument("--relevance-threshold", type=float, default=0.2)
    parser.add_argument("--ndcg-k", type=int, default=None)
    parser.add_argument("--group-col", default="strategy")
    parser.add_argument(
        "--write-default-gt",
        default=None,
        help="내장 ground truth 10개를 CSV로 저장할 경로",
    )
    args = parser.parse_args()

    if args.write_default_gt:
        gt_df = make_default_ground_truth_dataframe()
        gt_df.to_csv(args.write_default_gt, index=False, encoding="utf-8-sig")
        print("기본 ground truth 저장 완료:", args.write_default_gt)

    input_df = _load_input(args.input, args.sheet)
    evaluated = evaluate_retrieval_dataframe(
        input_df,
        question_col=args.question_col,
        ground_truth_col=args.ground_truth_col,
        contexts_col=args.contexts_col,
        relevance_threshold=args.relevance_threshold,
        ndcg_k=args.ndcg_k,
    )
    summary = summarize_retrieval(evaluated, group_col=args.group_col)

    evaluated.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary.to_csv(args.summary, index=False, encoding="utf-8-sig")

    print("평가 결과 저장 완료:", args.output)
    print("요약 결과 저장 완료:", args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
