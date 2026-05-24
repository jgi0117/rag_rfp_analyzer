"""
experiments/run_experiments.py
모든 embedding 실험을 일괄 실행하는 일반화 스크립트

실행 예시:
  # 실험 1개
  python experiments/run_experiments.py \
    --configs configs/experiments/fixed_600_100_top4_openai_small.yaml

  # 실험 여러 개 (순차 실행)
  python experiments/run_experiments.py \
    --configs configs/experiments/fixed_600_100_top4_openai_small.yaml \
             configs/experiments/markdown_600_100_top4_openai_small.yaml \
             configs/experiments/semantic_600_100_top4_openai_small.yaml

  # 실험 폴더 전체 자동 탐색
  python experiments/run_experiments.py --config_dir configs/experiments/

W&B 기록 구조:
  wandb.init()        실험 config 전체를 하이퍼파라미터로 등록
  wandb.log(step=N)   파이프라인 단계별 소요 시간 + 청킹 통계
  run.summary         최종 평가 지표 4개  (대시보드 실험 비교 컬럼)
  wandb.Table         쿼리별 상세 결과  (드릴다운용)
  wandb.Artifact      chunks CSV, evaluation CSV  (버전 관리)
  비교 테이블         모든 실험 완료 후 전체 결과 summary Table 기록
"""

import argparse
import csv
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import wandb
import yaml

from src.preprocessing import loader, cleaner
from src.evaluation.retrieval import (
    evaluate_retrieval_dataframe,
    make_default_ground_truth_dataframe,
    summarize_retrieval,
    save_results_csv,
)


# ── config 로드 ───────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """
    실험 yaml을 읽고, base_config: true 이면 configs/base.yaml 과 병합합니다.
    실험 yaml의 값이 base.yaml을 덮어씁니다.
    """
    with open(config_path, encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    if exp_cfg.get("base_config"):
        with open("configs/base.yaml", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        cfg = {**base_cfg, **exp_cfg}
    else:
        cfg = exp_cfg

    return cfg


# ── Chroma 저장 & search_fn 생성 ─────────────────────────────────────────────

def build_chroma_search_fn(chunks: list, cfg: dict):
    """
    Chroma DB에 chunk를 저장하고, 쿼리 → chunk_id 리스트를 반환하는 search_fn을 생성합니다.
    embedding provider에 따라 OpenAI / HuggingFace embedding function을 선택합니다.
    """
    import chromadb
    from chromadb.utils import embedding_functions

    provider = cfg["embedding"]["provider"]
    model    = cfg["embedding"]["model"]

    # ── embedding function 선택 ───────────────────────────────
    if provider == "openai":
        ef = embedding_functions.OpenAIEmbeddingFunction(
            model_name=model
        )
    elif provider == "huggingface":
        ef = embedding_functions.HuggingFaceEmbeddingFunction(
            model_name=model
        )
    else:
        raise ValueError(
            f"지원하지 않는 embedding provider: '{provider}'\n"
            f"사용 가능: openai | huggingface"
        )

    # ── Chroma 컬렉션 생성 & 저장 ────────────────────────────
    persist_dir = cfg["retrieval"]["persist_directory"]
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)

    # 동일 strategy_name으로 재실행 시 기존 컬렉션 초기화
    collection_name = cfg["output"]["strategy_name"]
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    col = client.get_or_create_collection(
        collection_name,
        embedding_function=ef
    )

    # 배치 단위로 upsert (Chroma 기본 limit: 5461)
    BATCH = 500
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        col.add(
            ids       = [c.chunk_id for c in batch],
            documents = [c.text     for c in batch],
            metadatas = [c.metadata for c in batch],
        )

    top_k = cfg["retrieval"]["top_k"]

    def search_fn(query: str) -> list[str]:
        results = col.query(
            query_texts=[query],
            n_results=top_k,
        )
        return results["ids"][0]

    return search_fn


# ── Artifact 저장 헬퍼 ───────────────────────────────────────────────────────

def _log_chunks_artifact(
    chunks: list, strategy_name: str, chunk_results_path: str
) -> None:
    """chunks CSV를 W&B Artifact로 등록합니다."""
    path = Path(chunk_results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["chunk_id", "text", "strategy", "source"]
        )
        writer.writeheader()
        for c in chunks:
            writer.writerow({
                "chunk_id": c.chunk_id,
                "text":     c.text,
                "strategy": c.metadata.get("strategy", ""),
                "source":   c.metadata.get("source", ""),
            })

    artifact = wandb.Artifact(
        name     = f"{strategy_name}-chunks",
        type     = "chunks",
        metadata = {
            "num_chunks": len(chunks),
            "splitter":   chunks[0].metadata.get("strategy", "") if chunks else "",
        },
    )
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)
    print(f"  [chunks] 저장 완료: {path.name}  ({len(chunks)}개)")


def _log_eval_artifact(result_df, summary, cfg: dict) -> None:
    """evaluation CSV 2개를 W&B Artifact로 등록합니다."""
    strategy_name = cfg["output"]["strategy_name"]
    result_path   = cfg["output"]["retrieval_eval_results"]
    summary_path  = cfg["output"]["retrieval_eval_summary"]

    save_results_csv(result_df, summary, result_path, summary_path)

    artifact = wandb.Artifact(
        name     = f"{strategy_name}-evaluation",
        type     = "evaluation",
        metadata = asdict(summary),
    )
    artifact.add_file(result_path)
    artifact.add_file(summary_path)
    wandb.log_artifact(artifact)


# ── 단일 실험 실행 ────────────────────────────────────────────────────────────

def run_single_experiment(config_path: str, wandb_project: str, wandb_entity: str) -> dict | None:
    """
    config_path 하나에 대해 전체 파이프라인을 실행합니다.
    성공 시 summary dict를 반환하고, 실패 시 None을 반환합니다.

    반환 예:
      {
        "strategy_name":     "markdown_600_100_top4_openai_small",
        "splitter":          "markdown",
        "embedding_model":   "text-embedding-3-small",
        "context_recall":    0.75,
        "context_precision": 0.50,
        "mrr":               0.83,
        "ndcg":              0.80,
        "num_chunks":        142,
        "config_file":       "markdown_600_100_top4_openai_small.yaml",
        "status":            "success",
      }
    """
    cfg           = load_config(config_path)
    strategy_name = cfg["output"]["strategy_name"]
    project_name  = Path(cfg["path"]["raw_pdf_file"]).stem

    # W&B project / entity는 CLI 인자 우선, 없으면 base.yaml 값 사용
    _project = wandb_project or cfg["wandb"]["project"]
    _entity  = wandb_entity  or cfg["wandb"]["entity"]

    run = wandb.init(
        project  = _project,
        entity   = _entity,
        name     = strategy_name,
        config   = cfg,
        tags     = [
            cfg["preprocessing"]["splitter"],
            cfg["embedding"]["model"],
            cfg["embedding"]["provider"],
        ],
        reinit   = True,   # 한 프로세스에서 여러 run을 순차 실행할 때 필요
    )

    print(f"\n{'='*60}")
    print(f"  실험 시작 : {strategy_name}")
    print(f"  config    : {config_path}")
    print(f"  splitter  = {cfg['preprocessing']['splitter']}")
    print(f"  embedding = {cfg['embedding']['model']}")
    print(f"  top_k     = {cfg['retrieval']['top_k']}")
    print(f"  W&B run   : {run.url}")
    print(f"{'='*60}\n")

    result_row = {
        "strategy_name":   strategy_name,
        "splitter":        cfg["preprocessing"]["splitter"],
        "embedding_model": cfg["embedding"]["model"],
        "config_file":     Path(config_path).name,
        "status":          "failed",
    }

    try:
        step = 0

        # ── [1] PDF → Markdown 파싱 ──────────────────────────
        print("[1/5] PDF 파싱 중 (loader.py)")
        t0  = time.time()
        doc = loader.extract_pdf(cfg["path"]["raw_pdf_file"])
        elapsed = round(time.time() - t0, 3)
        wandb.log(
            {"pipeline/pages": doc.pages, "pipeline/load_time_s": elapsed},
            step=step,
        )
        step += 1
        print(f"  → {doc.pages}p  ({elapsed}s)")

        # ── [2] 청킹 ─────────────────────────────────────────
        print(f"\n[2/5] 청킹 수행 중  splitter={cfg['preprocessing']['splitter']}")
        t0     = time.time()
        chunks = cleaner.get_chunks(doc.markdown, project_name, cfg)
        avg_len = sum(len(c.text) for c in chunks) / max(len(chunks), 1)
        elapsed = round(time.time() - t0, 3)
        wandb.log(
            {
                "pipeline/num_chunks":      len(chunks),
                "pipeline/avg_chunk_len":   round(avg_len, 1),
                "pipeline/chunking_time_s": elapsed,
            },
            step=step,
        )
        step += 1
        print(f"  → {len(chunks)}개 청크  (평균 {avg_len:.0f}자,  {elapsed}s)")

        _log_chunks_artifact(chunks, strategy_name, cfg["output"]["chunk_results"])

        # ── [3] 임베딩 & Chroma DB 저장 ──────────────────────
        print("\n[3/5] 임베딩 생성 및 Chroma DB 저장 중")
        t0        = time.time()
        search_fn = build_chroma_search_fn(chunks, cfg)
        elapsed   = round(time.time() - t0, 3)
        wandb.log({"pipeline/embed_store_time_s": elapsed}, step=step)
        step += 1
        print(f"  → DB 저장 완료  ({elapsed}s)")

        # ── [4] Top-k 검색 ────────────────────────────────────
        print("\n[4/5] Top-k 검색 중")
        t0        = time.time()
        gt_df     = make_default_ground_truth_dataframe()
        result_df = evaluate_retrieval_dataframe(
            gt_df, search_fn, top_k=cfg["retrieval"]["top_k"]
        )
        elapsed = round(time.time() - t0, 3)
        wandb.log({"pipeline/search_time_s": elapsed}, step=step)
        step += 1

        # ── [5] Retrieval 평가 ────────────────────────────────
        print("\n[5/5] Retrieval 평가 중 (retrieval.py)")
        summary = summarize_retrieval(result_df, strategy_name)

        per_query_table = wandb.Table(dataframe=result_df[[
            "query", "recall", "precision", "rr", "ndcg",
            "retrieved_ids", "relevant_ids",
        ]])
        wandb.log(
            {
                "eval/context_recall":    summary.context_recall,
                "eval/context_precision": summary.context_precision,
                "eval/mrr":               summary.mrr,
                "eval/ndcg":              summary.ndcg,
                "eval/per_query_table":   per_query_table,
            },
            step=step,
        )

        run.summary["eval/context_recall"]    = summary.context_recall
        run.summary["eval/context_precision"] = summary.context_precision
        run.summary["eval/mrr"]               = summary.mrr
        run.summary["eval/ndcg"]              = summary.ndcg

        _log_eval_artifact(result_df, summary, cfg)

        print(
            f"\n  ✅ 결과  "
            f"recall={summary.context_recall:.4f}  "
            f"precision={summary.context_precision:.4f}  "
            f"MRR={summary.mrr:.4f}  "
            f"nDCG={summary.ndcg:.4f}"
        )
        print(f"  W&B 대시보드: {run.url}")

        result_row.update({
            "context_recall":    summary.context_recall,
            "context_precision": summary.context_precision,
            "mrr":               summary.mrr,
            "ndcg":              summary.ndcg,
            "num_chunks":        len(chunks),
            "status":            "success",
        })
        return result_row

    except Exception:
        print(f"\n  ❌ 실험 실패: {strategy_name}")
        traceback.print_exc()
        return result_row

    finally:
        wandb.finish()


# ── 전체 실험 완료 후 비교 테이블 W&B 기록 ───────────────────────────────────

def log_comparison_table(all_results: list[dict], cfg: dict) -> None:
    """
    모든 실험 결과를 하나의 W&B run에 비교 테이블로 기록합니다.
    이 run은 대시보드에서 실험 간 성능을 한눈에 비교하는 용도입니다.
    """
    import pandas as pd

    if not any(r["status"] == "success" for r in all_results):
        print("\n  성공한 실험이 없어 비교 테이블을 생성하지 않습니다.")
        return

    run = wandb.init(
        project = cfg["wandb"]["project"],
        entity  = cfg["wandb"]["entity"],
        name    = "experiment_comparison",
        job_type = "comparison",
        reinit  = True,
    )

    success_results = [r for r in all_results if r["status"] == "success"]
    df = pd.DataFrame(success_results)

    # 지표별 최고 실험 하이라이트
    best_recall    = df.loc[df["context_recall"].idxmax(),    "strategy_name"]
    best_precision = df.loc[df["context_precision"].idxmax(), "strategy_name"]
    best_mrr       = df.loc[df["mrr"].idxmax(),               "strategy_name"]
    best_ndcg      = df.loc[df["ndcg"].idxmax(),              "strategy_name"]

    run.summary["best/recall_strategy"]    = best_recall
    run.summary["best/precision_strategy"] = best_precision
    run.summary["best/mrr_strategy"]       = best_mrr
    run.summary["best/ndcg_strategy"]      = best_ndcg

    comparison_table = wandb.Table(dataframe=df[[
        "strategy_name", "splitter", "embedding_model",
        "context_recall", "context_precision", "mrr", "ndcg",
        "num_chunks", "status",
    ]])
    wandb.log({"comparison/all_experiments": comparison_table})

    wandb.finish()

    # 콘솔 출력
    print(f"\n{'='*60}")
    print("  전체 실험 결과 비교")
    print(f"{'='*60}")
    print(df[[
        "strategy_name", "context_recall", "context_precision", "mrr", "ndcg"
    ]].to_string(index=False))
    print(f"\n  🏆 Best recall    : {best_recall}")
    print(f"  🏆 Best precision : {best_precision}")
    print(f"  🏆 Best MRR       : {best_mrr}")
    print(f"  🏆 Best nDCG      : {best_ndcg}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG embedding 실험 일괄 실행 스크립트"
    )

    # ── 실험 config 지정 방식 (둘 중 하나 사용) ──────────────
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--configs",
        nargs="+",
        metavar="CONFIG_YAML",
        help="실험 config yaml 경로를 하나 이상 지정합니다.\n"
             "예: --configs configs/experiments/fixed_600_100_top4_openai_small.yaml "
             "configs/experiments/markdown_600_100_top4_openai_small.yaml",
    )
    config_group.add_argument(
        "--config_dir",
        metavar="DIR",
        help="폴더 안의 모든 *.yaml을 자동으로 탐색합니다.\n"
             "예: --config_dir configs/experiments/",
    )

    # ── W&B override (base.yaml보다 우선) ────────────────────
    parser.add_argument(
        "--wandb_project",
        default=None,
        help="W&B 프로젝트명 (지정 시 base.yaml의 wandb.project를 덮어씁니다)",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="W&B 엔티티명 (지정 시 base.yaml의 wandb.entity를 덮어씁니다)",
    )

    # ── 실행 제어 ─────────────────────────────────────────────
    parser.add_argument(
        "--skip_failed",
        action="store_true",
        default=True,
        help="개별 실험 실패 시 전체를 중단하지 않고 다음 실험으로 넘어갑니다 (기본값: True)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="config 파일 목록만 출력하고 실제 실험은 실행하지 않습니다",
    )

    args = parser.parse_args()

    # ── 실험 config 파일 목록 수집 ────────────────────────────
    if args.configs:
        config_paths = [Path(p) for p in args.configs]
    else:
        config_dir   = Path(args.config_dir)
        config_paths = sorted(config_dir.glob("*.yaml"))

    if not config_paths:
        print("  ❌ 실험 config 파일을 찾지 못했습니다.")
        return

    print(f"\n  실행 예정 실험 수: {len(config_paths)}개")
    for i, p in enumerate(config_paths, 1):
        print(f"    [{i:02d}] {p}")

    if args.dry_run:
        print("\n  --dry_run 모드: 실제 실험을 실행하지 않습니다.")
        return

    # ── W&B 로그인 (전체 실험 시작 전 1회) ───────────────────
    wandb.login()

    # ── 순차 실험 실행 ────────────────────────────────────────
    all_results = []
    total = len(config_paths)

    for idx, config_path in enumerate(config_paths, 1):
        print(f"\n\n{'#'*60}")
        print(f"  [{idx}/{total}] 실험 실행: {config_path.name}")
        print(f"{'#'*60}")

        result = run_single_experiment(
            config_path  = str(config_path),
            wandb_project = args.wandb_project,
            wandb_entity  = args.wandb_entity,
        )

        if result is not None:
            all_results.append(result)

        if result and result["status"] == "failed" and not args.skip_failed:
            print(f"\n  --skip_failed=False: 실험 중단")
            break

    # ── 전체 비교 테이블 기록 ─────────────────────────────────
    if len(all_results) > 1:
        # base.yaml에서 W&B 설정을 읽어 비교 run에 사용
        with open("configs/base.yaml", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        if args.wandb_project:
            base_cfg["wandb"]["project"] = args.wandb_project
        if args.wandb_entity:
            base_cfg["wandb"]["entity"] = args.wandb_entity

        log_comparison_table(all_results, base_cfg)

    # ── 최종 요약 ─────────────────────────────────────────────
    success = sum(1 for r in all_results if r["status"] == "success")
    failed  = sum(1 for r in all_results if r["status"] == "failed")
    print(f"\n\n{'='*60}")
    print(f"  실험 완료  성공: {success}개  실패: {failed}개  (총 {total}개)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
