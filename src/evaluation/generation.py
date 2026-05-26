"""
generation_eval_metrics.py

RAG Generation 평가용 공통 함수 모음
- Faithfulness: 생성 답변이 retrieved_context에 근거하는지 평가
- Answer Relevance: 생성 답변이 question에 적합한지 평가
- BLEU: generated_answer와 ground_truth_answer의 N-gram 정밀도
- ROUGE: ground_truth_answer 기준 재현율 계열 점수
- F1 score: 토큰 overlap 기반 정밀도/재현율 조화평균
- Metadata Match: 메타데이터 컬럼 값이 생성 답변에 포함/일치하는지 참고 점수

사용 전제 CSV 컬럼 예시:
strategy, question_id, question, ground_truth_answer, retrieved_context, generated_answer,
공고 번호, 공고 차수, 사업명, 사업 금액, 발주 기관, 공개 일자,
입찰 참여 시작일, 입찰 참여 마감일, 사업 요약, 파일형식, 파일명
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# =========================================================
# 0. 기본 유틸
# =========================================================

def normalize_text(text: Any) -> str:
    """공백/개행을 정리하고 문자열로 변환한다."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: Any) -> List[str]:
    """
    한국어/영어/숫자를 단순 토큰화한다.
    형태소 분석기를 쓰지 않는 공통 baseline용 tokenizer.
    """
    text = normalize_text(text).lower()
    return re.findall(r"[가-힣]+|[a-zA-Z]+|\d+(?:\.\d+)?", text)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


# =========================================================
# 1. BLEU
# =========================================================

def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    generated_answer: str,
    ground_truth_answer: str,
    max_n: int = 4,
    smooth: float = 1.0,
) -> float:
    """
    BLEU 점수 계산.
    - generated_answer: 모델 답변
    - ground_truth_answer: 기준 정답
    - max_n: 1~4 gram까지 사용
    - smooth: 0점 방지를 위한 smoothing 값
    """
    pred_tokens = tokenize(generated_answer)
    ref_tokens = tokenize(ground_truth_answer)

    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_counts = _ngram_counts(pred_tokens, n)
        ref_counts = _ngram_counts(ref_tokens, n)

        overlap = sum(min(count, ref_counts[ng]) for ng, count in pred_counts.items())
        total = sum(pred_counts.values())
        precision_n = (overlap + smooth) / (total + smooth)
        precisions.append(precision_n)

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    # brevity penalty
    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    bp = 1.0 if pred_len > ref_len else math.exp(1 - safe_div(ref_len, pred_len))

    return round(bp * geo_mean, 4)


# =========================================================
# 2. ROUGE
# =========================================================

def compute_rouge_n(generated_answer: str, ground_truth_answer: str, n: int = 1) -> float:
    """
    ROUGE-N Recall 계산.
    기준 정답의 n-gram이 생성 답변에 얼마나 포함되었는지 측정한다.
    """
    pred_tokens = tokenize(generated_answer)
    ref_tokens = tokenize(ground_truth_answer)

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = _ngram_counts(pred_tokens, n)
    ref_counts = _ngram_counts(ref_tokens, n)

    overlap = sum(min(count, pred_counts[ng]) for ng, count in ref_counts.items())
    total_ref = sum(ref_counts.values())

    return round(safe_div(overlap, total_ref), 4)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest Common Subsequence 길이 계산."""
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            temp = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def compute_rouge_l(generated_answer: str, ground_truth_answer: str) -> float:
    """
    ROUGE-L Recall 계산.
    기준 정답 토큰열 대비 생성 답변과의 LCS 비율을 측정한다.
    """
    pred_tokens = tokenize(generated_answer)
    ref_tokens = tokenize(ground_truth_answer)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, pred_tokens)
    return round(safe_div(lcs, len(ref_tokens)), 4)


def compute_rouge(generated_answer: str, ground_truth_answer: str) -> Dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L을 함께 반환한다."""
    return {
        "rouge_1": compute_rouge_n(generated_answer, ground_truth_answer, n=1),
        "rouge_2": compute_rouge_n(generated_answer, ground_truth_answer, n=2),
        "rouge_l": compute_rouge_l(generated_answer, ground_truth_answer),
    }


# =========================================================
# 3. Token F1
# =========================================================

def compute_token_f1(generated_answer: str, ground_truth_answer: str) -> Dict[str, float]:
    """
    생성 답변과 기준 정답의 토큰 overlap 기반 Precision, Recall, F1 계산.
    """
    pred_tokens = tokenize(generated_answer)
    ref_tokens = tokenize(ground_truth_answer)

    if not pred_tokens or not ref_tokens:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())

    precision = safe_div(overlap, len(pred_tokens))
    recall = safe_div(overlap, len(ref_tokens))
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(f1, 4),
    }


# =========================================================
# 4. Faithfulness / Answer Relevance
# =========================================================

def compute_faithfulness_heuristic(generated_answer: str, retrieved_context: str) -> float:
    """
    LLM judge 없이 사용할 수 있는 간단한 Faithfulness proxy.
    생성 답변의 주요 토큰이 retrieved_context 안에 얼마나 존재하는지 계산한다.

    주의:
    - 진짜 의미의 hallucination 판별은 LLM-as-judge 또는 사람 평가가 더 적합하다.
    - 이 함수는 팀 공통 baseline 점수로 사용한다.
    """
    answer_tokens = tokenize(generated_answer)
    context_tokens = set(tokenize(retrieved_context))

    if not answer_tokens or not context_tokens:
        return 0.0

    supported = sum(1 for token in answer_tokens if token in context_tokens)
    return round(safe_div(supported, len(answer_tokens)), 4)


def compute_answer_relevance_heuristic(generated_answer: str, question: str) -> float:
    """
    LLM judge 없이 사용할 수 있는 간단한 Answer Relevance proxy.
    질문의 핵심 토큰이 생성 답변 안에 얼마나 반영되었는지 계산한다.
    """
    question_tokens = tokenize(question)
    answer_tokens = set(tokenize(generated_answer))

    if not question_tokens or not answer_tokens:
        return 0.0

    matched = sum(1 for token in question_tokens if token in answer_tokens)
    return round(safe_div(matched, len(question_tokens)), 4)


@dataclass
class LLMJudgeResult:
    score: float
    reason: str = ""


def build_faithfulness_prompt(question: str, retrieved_context: str, generated_answer: str) -> str:
    """LLM-as-judge용 Faithfulness 평가 프롬프트."""
    return f"""
당신은 RAG 답변 평가자입니다.
아래 답변이 검색된 컨텍스트에 근거해서만 작성되었는지 평가하세요.

[평가 기준]
1점: 컨텍스트에 없는 내용을 지어냈거나 핵심 사실이 맞지 않음
3점: 일부는 근거가 있으나 불명확하거나 누락/추론이 섞임
5점: 답변의 핵심 내용이 모두 컨텍스트에 근거함

[질문]
{question}

[검색된 컨텍스트]
{retrieved_context}

[생성 답변]
{generated_answer}

반드시 아래 형식으로만 답하세요.
score: 1~5 숫자
reason: 한 줄 평가 이유
""".strip()


def build_answer_relevance_prompt(question: str, generated_answer: str) -> str:
    """LLM-as-judge용 Answer Relevance 평가 프롬프트."""
    return f"""
당신은 RAG 답변 평가자입니다.
아래 답변이 사용자의 질문 의도에 얼마나 적합한지 평가하세요.

[평가 기준]
1점: 질문과 거의 관련 없는 답변
3점: 일부 관련 있으나 핵심 요구를 충분히 만족하지 못함
5점: 질문 의도에 직접적으로 답하고 필요한 정보를 충분히 포함함

[질문]
{question}

[생성 답변]
{generated_answer}

반드시 아래 형식으로만 답하세요.
score: 1~5 숫자
reason: 한 줄 평가 이유
""".strip()


def parse_llm_judge_response(response: str) -> LLMJudgeResult:
    """LLM judge 응답에서 score와 reason을 파싱한다."""
    response = normalize_text(response)
    score_match = re.search(r"score\s*:\s*([1-5](?:\.0)?)", response, re.I)
    reason_match = re.search(r"reason\s*:\s*(.*)", response, re.I)

    score = float(score_match.group(1)) if score_match else 0.0
    reason = reason_match.group(1).strip() if reason_match else ""
    return LLMJudgeResult(score=round(score / 5, 4), reason=reason)


def compute_faithfulness_llm(
    question: str,
    retrieved_context: str,
    generated_answer: str,
    judge_fn: Callable[[str], str],
) -> LLMJudgeResult:
    """
    외부 LLM 호출 함수를 주입받아 Faithfulness를 0~1 점수로 반환한다.
    judge_fn은 prompt 문자열을 입력받아 LLM 응답 문자열을 반환해야 한다.
    """
    prompt = build_faithfulness_prompt(question, retrieved_context, generated_answer)
    return parse_llm_judge_response(judge_fn(prompt))


def compute_answer_relevance_llm(
    question: str,
    generated_answer: str,
    judge_fn: Callable[[str], str],
) -> LLMJudgeResult:
    """
    외부 LLM 호출 함수를 주입받아 Answer Relevance를 0~1 점수로 반환한다.
    judge_fn은 prompt 문자열을 입력받아 LLM 응답 문자열을 반환해야 한다.
    """
    prompt = build_answer_relevance_prompt(question, generated_answer)
    return parse_llm_judge_response(judge_fn(prompt))


# =========================================================
# 5. 메타데이터 참고 평가
# =========================================================

DEFAULT_METADATA_COLUMNS = [
    "공고 번호",
    "공고 차수",
    "사업명",
    "사업 금액",
    "발주 기관",
    "공개 일자",
    "입찰 참여 시작일",
    "입찰 참여 마감일",
    "사업 요약",
    "파일형식",
    "파일명",
]


def compute_metadata_match_score(
    generated_answer: str,
    row: Dict[str, Any],
    metadata_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    참고용 메타데이터 일치 점수.
    row에 들어있는 메타데이터 값이 generated_answer에 포함되어 있는지 확인한다.

    반환:
    - metadata_match_score: 일치한 메타데이터 비율
    - metadata_matched_fields: 일치한 컬럼명
    - metadata_checked_fields: 실제 체크한 컬럼명
    """
    metadata_columns = list(metadata_columns or DEFAULT_METADATA_COLUMNS)
    answer = normalize_text(generated_answer)

    checked = []
    matched = []

    for col in metadata_columns:
        value = normalize_text(row.get(col, ""))
        if not value or value.lower() == "nan":
            continue

        # 너무 긴 사업 요약은 토큰 일부만 참고하도록 처리
        if col == "사업 요약" and len(value) > 50:
            value_tokens = tokenize(value)[:8]
            is_match = any(token in tokenize(answer) for token in value_tokens)
        else:
            is_match = value in answer

        checked.append(col)
        if is_match:
            matched.append(col)

    return {
        "metadata_match_score": round(safe_div(len(matched), len(checked)), 4),
        "metadata_matched_fields": ", ".join(matched),
        "metadata_checked_fields": ", ".join(checked),
    }


# =========================================================
# 6. 단일 row / 전체 DataFrame 평가
# =========================================================

def evaluate_generation_row(
    row: Dict[str, Any],
    question_col: str = "question",
    ground_truth_col: str = "ground_truth_answer",
    context_col: str = "retrieved_context",
    answer_col: str = "generated_answer",
    metadata_columns: Optional[Sequence[str]] = None,
    judge_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """
    하나의 generation 결과 row를 평가한다.
    judge_fn이 있으면 LLM-as-judge 점수를 사용하고, 없으면 heuristic 점수를 사용한다.
    """
    question = normalize_text(row.get(question_col, ""))
    ground_truth = normalize_text(row.get(ground_truth_col, ""))
    context = normalize_text(row.get(context_col, ""))
    answer = normalize_text(row.get(answer_col, ""))

    rouge_scores = compute_rouge(answer, ground_truth)
    f1_scores = compute_token_f1(answer, ground_truth)
    metadata_scores = compute_metadata_match_score(answer, row, metadata_columns)

    if judge_fn is not None:
        faith = compute_faithfulness_llm(question, context, answer, judge_fn)
        relevance = compute_answer_relevance_llm(question, answer, judge_fn)
        faithfulness = faith.score
        answer_relevance = relevance.score
        faithfulness_reason = faith.reason
        answer_relevance_reason = relevance.reason
    else:
        faithfulness = compute_faithfulness_heuristic(answer, context)
        answer_relevance = compute_answer_relevance_heuristic(answer, question)
        faithfulness_reason = "heuristic: generated_answer token support ratio in retrieved_context"
        answer_relevance_reason = "heuristic: question token coverage ratio in generated_answer"

    return {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "bleu": compute_bleu(answer, ground_truth),
        **rouge_scores,
        **f1_scores,
        **metadata_scores,
        "faithfulness_reason": faithfulness_reason,
        "answer_relevance_reason": answer_relevance_reason,
    }


def evaluate_generation_dataframe(
    df,
    question_col: str = "question",
    ground_truth_col: str = "ground_truth_answer",
    context_col: str = "retrieved_context",
    answer_col: str = "generated_answer",
    metadata_columns: Optional[Sequence[str]] = None,
    judge_fn: Optional[Callable[[str], str]] = None,
):
    """
    pandas DataFrame 전체를 평가한다.
    pandas는 함수 내부에서 import하지 않으므로, 호출 측에서 pandas DataFrame을 넘기면 된다.
    """
    scores = []
    for _, row in df.iterrows():
        scores.append(
            evaluate_generation_row(
                row.to_dict(),
                question_col=question_col,
                ground_truth_col=ground_truth_col,
                context_col=context_col,
                answer_col=answer_col,
                metadata_columns=metadata_columns,
                judge_fn=judge_fn,
            )
        )

    import pandas as pd

    score_df = pd.DataFrame(scores)
    return pd.concat([df.reset_index(drop=True), score_df], axis=1)


def summarize_by_strategy(
    evaluated_df,
    strategy_col: str = "strategy",
    metric_cols: Optional[Sequence[str]] = None,
):
    """전략별 평균 점수 요약표를 생성한다."""
    import pandas as pd

    metric_cols = list(metric_cols or [
        "faithfulness",
        "answer_relevance",
        "bleu",
        "rouge_1",
        "rouge_2",
        "rouge_l",
        "token_precision",
        "token_recall",
        "token_f1",
        "metadata_match_score",
    ])
    if "generation_seconds" in evaluated_df.columns and "generation_seconds" not in metric_cols:
        metric_cols.append("generation_seconds")

    summary = (
        evaluated_df
        .groupby(strategy_col)[metric_cols]
        .mean(numeric_only=True)
        .round(4)
        .reset_index()
    )
    return summary


# =========================================================
# 7. 실행 예시
# =========================================================

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="RAG generation 결과 CSV/XLSX 평가")
    parser.add_argument("--input", required=True, help="generation 결과 파일 경로(csv/xlsx)")
    parser.add_argument("--output", default="generation_eval_result.csv", help="평가 결과 저장 경로")
    parser.add_argument("--summary", default="generation_eval_summary.csv", help="전략별 요약 저장 경로")
    parser.add_argument("--sheet", default=0, help="xlsx 입력 시 sheet 이름 또는 index")
    args = parser.parse_args()

    if args.input.lower().endswith(".xlsx"):
        sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
        input_df = pd.read_excel(args.input, sheet_name=sheet)
    else:
        input_df = pd.read_csv(args.input)

    evaluated = evaluate_generation_dataframe(input_df)
    summary = summarize_by_strategy(evaluated)

    evaluated.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary.to_csv(args.summary, index=False, encoding="utf-8-sig")

    print("Saved evaluation result:", args.output)
    print("Saved summary result:", args.summary)
