"""
Metadata-aware retrieval evaluation for multiple RFP files.

This module keeps the original retrieval metrics from `retrieval.py`, but adds
helpers for `data_list.csv` style datasets where each row represents one RFP
file with notice metadata and extracted text.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from .retrieval import (
    compute_chunk_relevance,
    compute_context_precision,
    compute_context_recall,
    compute_mrr,
    compute_ndcg,
    normalize_text,
    parse_retrieved_contexts,
    summarize_retrieval as summarize_retrieval_v1,
)


DEFAULT_METADATA_PATH = "rag_rfp_analyzer/data/data_list.csv"
LOCAL_METADATA_FALLBACK_PATH = "data/raw/data_list.csv"

NOTICE_NUMBER_COL = "공고 번호"
NOTICE_ROUND_COL = "공고 차수"
PROJECT_NAME_COL = "사업명"
BUDGET_COL = "사업 금액"
AGENCY_COL = "발주 기관"
PUBLISHED_AT_COL = "공개 일자"
BID_START_COL = "입찰 참여 시작일"
BID_END_COL = "입찰 참여 마감일"
SUMMARY_COL = "사업 요약"
FILE_TYPE_COL = "파일형식"
FILE_NAME_COL = "파일명"
SOURCE_TEXT_COL = "텍스트"

METADATA_COLUMNS = [
    PROJECT_NAME_COL,
    BUDGET_COL,
    AGENCY_COL,
    PUBLISHED_AT_COL,
    BID_START_COL,
    BID_END_COL,
    SUMMARY_COL,
]

EVALUATION_FIELD_MAP = [
    ("project_name", PROJECT_NAME_COL),
    ("budget", BUDGET_COL),
    ("agency", AGENCY_COL),
    ("published_at", PUBLISHED_AT_COL),
    ("bid_start", BID_START_COL),
    ("bid_end", BID_END_COL),
    ("summary", SUMMARY_COL),
]


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    text = normalize_text(value)
    return not text or text.lower() in {"nan", "nat", "none"}


def format_metadata_value(value: Any) -> str:
    """Normalize pandas/numeric metadata values into stable comparison text."""
    if is_empty(value):
        return ""

    text = normalize_text(value)
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]

    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return text

    if number.is_integer():
        return str(int(number))
    return text


def resolve_metadata_path(path: str = DEFAULT_METADATA_PATH) -> Path:
    metadata_path = Path(path)
    if metadata_path.exists():
        return metadata_path

    normalized_path = path.replace("\\", "/")
    if normalized_path == DEFAULT_METADATA_PATH or normalized_path.endswith(f"/{DEFAULT_METADATA_PATH}"):
        fallback = Path(LOCAL_METADATA_FALLBACK_PATH)
        if fallback.exists():
            return fallback

    return metadata_path


def load_metadata_dataframe(path: str = DEFAULT_METADATA_PATH, sheet: Any = 0):
    import pandas as pd

    metadata_path = resolve_metadata_path(path)
    if metadata_path.suffix.lower() == ".xlsx":
        sheet_name = int(sheet) if str(sheet).isdigit() else sheet
        return pd.read_excel(metadata_path, sheet_name=sheet_name)

    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(metadata_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(metadata_path)


def metadata_row_to_ground_truth(row: dict[str, Any]) -> str:
    parts = []
    for column in METADATA_COLUMNS:
        value = format_metadata_value(row.get(column))
        if value:
            parts.append(f"{column}: {value}")
    return " ".join(parts)


def metadata_row_to_question(row: dict[str, Any], row_number: int) -> str:
    project_name = format_metadata_value(row.get(PROJECT_NAME_COL))
    file_name = format_metadata_value(row.get(FILE_NAME_COL))
    notice_number = format_metadata_value(row.get(NOTICE_NUMBER_COL))
    subject = project_name or file_name or notice_number or f"{row_number}번째 공고"
    return (
        f"{subject} 사업의 사업명, 사업 금액, 발주 기관, 공개 일자, "
        "입찰 참여 기간과 사업 요약을 알려주세요."
    )


def make_metadata_ground_truth_dataframe(
    metadata_path: str = DEFAULT_METADATA_PATH,
    *,
    limit: Optional[int] = None,
    include_source_text: bool = False,
    sheet: Any = 0,
):
    """Convert `data_list.csv` metadata rows into retrieval evaluation QA rows."""
    import pandas as pd

    metadata_df = load_metadata_dataframe(metadata_path, sheet=sheet)
    if limit is not None:
        metadata_df = metadata_df.head(limit)

    eval_rows = []
    for idx, row in metadata_df.reset_index(drop=True).iterrows():
        row_dict = row.to_dict()
        notice_number = format_metadata_value(row_dict.get(NOTICE_NUMBER_COL))

        eval_row = {
            "question_id": notice_number or idx + 1,
            "question": metadata_row_to_question(row_dict, idx + 1),
            "ground_truth": metadata_row_to_ground_truth(row_dict),
        }
        for field_name, column in EVALUATION_FIELD_MAP:
            eval_row[f"expected_{field_name}"] = format_metadata_value(row_dict.get(column))
        if include_source_text:
            eval_row["source_text"] = format_metadata_value(row_dict.get(SOURCE_TEXT_COL))
        eval_rows.append(eval_row)

    return pd.DataFrame(eval_rows)


def make_metadata_ground_truth_from_dataframe(
    metadata_df,
    *,
    include_source_text: bool = False,
):
    """Convert an already-loaded metadata dataframe into retrieval evaluation QA rows."""
    import pandas as pd

    eval_rows = []
    for idx, row in metadata_df.reset_index(drop=True).iterrows():
        row_dict = row.to_dict()
        notice_number = format_metadata_value(row_dict.get(NOTICE_NUMBER_COL))

        eval_row = {
            "question_id": notice_number or idx + 1,
            "question": metadata_row_to_question(row_dict, idx + 1),
            "ground_truth": metadata_row_to_ground_truth(row_dict),
        }
        for field_name, column in EVALUATION_FIELD_MAP:
            eval_row[f"expected_{field_name}"] = format_metadata_value(row_dict.get(column))
        if include_source_text:
            eval_row["source_text"] = format_metadata_value(row_dict.get(SOURCE_TEXT_COL))
        eval_rows.append(eval_row)

    return pd.DataFrame(eval_rows)


def contains_expected_text(contexts: Sequence[str], expected: Any) -> float:
    expected_text = format_metadata_value(expected).lower()
    if not expected_text:
        return 0.0
    joined_contexts = normalize_text(" ".join(contexts)).lower()
    return 1.0 if expected_text in joined_contexts else 0.0


def evaluate_retrieval_row_v2(
    row: dict[str, Any],
    question_col: str = "question",
    ground_truth_col: str = "ground_truth",
    contexts_col: str = "retrieved_contexts",
    relevance_threshold: float = 0.2,
    ndcg_k: Optional[int] = None,
) -> dict[str, Any]:
    contexts = parse_retrieved_contexts(row, contexts_col=contexts_col)
    ground_truth = normalize_text(row.get(ground_truth_col, ""))
    question = normalize_text(row.get(question_col, ""))
    relevance_scores = [compute_chunk_relevance(ground_truth, context) for context in contexts]

    first_relevant_rank = 0
    for rank, score in enumerate(relevance_scores, start=1):
        if score >= relevance_threshold:
            first_relevant_rank = rank
            break

    metadata_hits = {
        f"{field_name}_hit": contains_expected_text(contexts, row.get(f"expected_{field_name}"))
        for field_name, _ in EVALUATION_FIELD_MAP
    }
    available_hit_values = [
        metadata_hits[f"{field_name}_hit"]
        for field_name, _ in EVALUATION_FIELD_MAP
        if not is_empty(row.get(f"expected_{field_name}"))
    ]

    scores = {
        "context_recall": compute_context_recall(ground_truth, contexts),
        "context_precision": compute_context_precision(relevance_scores, relevance_threshold),
        "mrr": compute_mrr(relevance_scores, relevance_threshold),
        "ndcg": compute_ndcg(relevance_scores, ndcg_k),
        "metadata_hit_rate": (
            round(sum(available_hit_values) / len(available_hit_values), 4)
            if available_hit_values
            else 0.0
        ),
        "retrieved_count": len(contexts),
        "first_relevant_rank": first_relevant_rank,
        "chunk_relevance_scores": ", ".join(f"{score:.4f}" for score in relevance_scores),
        "question_for_eval": question,
    }
    scores.update(metadata_hits)
    return scores


def evaluate_retrieval_dataframe_v2(
    df,
    question_col: str = "question",
    ground_truth_col: str = "ground_truth",
    contexts_col: str = "retrieved_contexts",
    relevance_threshold: float = 0.2,
    ndcg_k: Optional[int] = None,
):
    import pandas as pd

    scores = [
        evaluate_retrieval_row_v2(
            row.to_dict(),
            question_col=question_col,
            ground_truth_col=ground_truth_col,
            contexts_col=contexts_col,
            relevance_threshold=relevance_threshold,
            ndcg_k=ndcg_k,
        )
        for _, row in df.iterrows()
    ]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(scores)], axis=1)


def summarize_retrieval_v2(
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
            "metadata_hit_rate",
            "project_name_hit",
            "budget_hit",
            "agency_hit",
            "published_at_hit",
            "bid_start_hit",
            "bid_end_hit",
            "summary_hit",
            "retrieved_count",
        ]
    )
    return summarize_retrieval_v1(evaluated_df, group_col=group_col, metric_cols=metric_cols)


def load_input_dataframe(path: str, sheet: Any = 0):
    import pandas as pd

    input_path = Path(path)
    if input_path.suffix.lower() == ".xlsx":
        sheet_name = int(sheet) if str(sheet).isdigit() else sheet
        return pd.read_excel(input_path, sheet_name=sheet_name)
    return pd.read_csv(input_path)


def write_csv(df, path: str) -> None:
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Metadata-aware RAG retrieval evaluation")
    parser.add_argument("--input", default=None, help="검색 결과 파일 경로(csv/xlsx)")
    parser.add_argument("--output", default="retrieval_v2_eval_result.csv", help="평가 결과 저장 경로")
    parser.add_argument("--summary", default="retrieval_v2_eval_summary.csv", help="요약 저장 경로")
    parser.add_argument("--sheet", default=0, help="xlsx 입력 시 sheet 이름 또는 index")
    parser.add_argument("--question-col", default="question")
    parser.add_argument("--ground-truth-col", default="ground_truth")
    parser.add_argument("--contexts-col", default="retrieved_contexts")
    parser.add_argument("--relevance-threshold", type=float, default=0.2)
    parser.add_argument("--ndcg-k", type=int, default=None)
    parser.add_argument("--group-col", default="strategy")
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH, help="공고 메타데이터 경로")
    parser.add_argument("--metadata-limit", type=int, default=None, help="메타데이터 변환 개수 제한")
    parser.add_argument(
        "--include-source-text",
        action="store_true",
        help="메타데이터 QA 저장 시 원문 텍스트 컬럼도 함께 포함",
    )
    parser.add_argument(
        "--write-metadata-gt",
        default=None,
        help="공고 메타데이터를 retrieval 평가용 QA CSV로 저장할 경로",
    )
    args = parser.parse_args()

    if args.write_metadata_gt:
        gt_df = make_metadata_ground_truth_dataframe(
            args.metadata,
            limit=args.metadata_limit,
            include_source_text=args.include_source_text,
            sheet=args.sheet,
        )
        write_csv(gt_df, args.write_metadata_gt)
        print("메타데이터 ground truth 저장 완료:", args.write_metadata_gt)

    if not args.input:
        if args.write_metadata_gt:
            return 0
        parser.error("--input 또는 --write-metadata-gt 중 하나가 필요합니다.")

    input_df = load_input_dataframe(args.input, args.sheet)
    evaluated = evaluate_retrieval_dataframe_v2(
        input_df,
        question_col=args.question_col,
        ground_truth_col=args.ground_truth_col,
        contexts_col=args.contexts_col,
        relevance_threshold=args.relevance_threshold,
        ndcg_k=args.ndcg_k,
    )
    summary = summarize_retrieval_v2(evaluated, group_col=args.group_col)

    write_csv(evaluated, args.output)
    write_csv(summary, args.summary)

    print("평가 결과 저장 완료:", args.output)
    print("요약 결과 저장 완료:", args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
