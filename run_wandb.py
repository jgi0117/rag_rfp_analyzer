"""
experiments/run_experiments.py
embedding + generation 전체 파이프라인을 일괄 실행하고 W&B에 기록하는 통합 스크립트

실행 예시:
  # 실험 1개
  python experiments/run_experiments.py \
    --configs configs/experiments/baseline.yaml

  # 실험 여러 개 (순차 실행)
  python experiments/run_experiments.py \
    --configs configs/experiments/baseline.yaml \
             configs/ablations/abl_splitter_semantic.yaml

  # 실험 폴더 전체 자동 탐색
  python experiments/run_experiments.py --config_dir configs/experiments/

  # 실험 진행자 지정
  python experiments/run_experiments.py \
    --configs configs/experiments/baseline.yaml \
    --experimenter 홍길동

W&B 기록 구조:
  wandb.init()             실험 config 전체를 하이퍼파라미터로 등록
  wandb.log(step=N)        파이프라인 단계별 소요 시간 + 지표
  run.summary              최종 평가 지표 (대시보드 실험 비교 컬럼)
  wandb.Table              쿼리별 상세 결과 + generation 결과 (드릴다운용)
  wandb.Artifact           chunks CSV, evaluation CSV (버전 관리)
  comparison run           모든 실험 완료 후 전체 결과 비교 테이블

W&B 기록 항목:
  meta/embedding_model      임베딩 모델명
  meta/embedding_provider   임베딩 provider
  meta/llm_model            LLM 모델명
  meta/llm_provider         LLM provider
  meta/chunking_strategy    청킹 전략
  meta/chunk_size           청크 크기
  meta/chunk_overlap        청크 오버랩
  meta/top_k                검색 top_k
  meta/temperature          생성 temperature
  meta/max_tokens           생성 max_tokens
  meta/experimenter         실험 진행자
  meta/config_file          사용한 config 파일명
  env/*                     OS, Python, GPU 등 실험 환경
  pipeline/*                단계별 소요 시간
  eval/*                    retrieval 정량 평가 지표 (recall, precision, mrr, ndcg)
  generation/*              generation 정량 평가 지표 + 응답 테이블
    generation/avg_generation_time_s  평균 생성 시간
    generation/avg_rouge1             평균 ROUGE-1
    generation/avg_rouge2             평균 ROUGE-2
    generation/avg_rougeL             평균 ROUGE-L
    generation/avg_exact_match        평균 Exact Match
    generation/avg_answer_length      평균 응답 길이
    generation/response_table         질문 | Ground Truth | 생성 응답 |
                                      generation_time | rouge1/2/L | exact_match
"""

import argparse
import csv
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import wandb
import yaml

from src.preprocessing import loader, cleaner
from src.evaluation.retrieval import (
    evaluate_retrieval_dataframe,
    make_default_ground_truth_dataframe,
    summarize_retrieval,
    save_results_csv,
)


# ── config 로드 ────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """
    실험 yaml을 읽고, base_config 키가 있으면 해당 경로의 yaml과 deep merge합니다.
    실험 yaml의 값이 base yaml을 덮어씁니다.
    """
    with open(config_path, encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    base_config_val = exp_cfg.get("base_config")
    if not base_config_val:
        return exp_cfg

    # base_config: true → configs/base.yaml 사용
    # base_config: "configs/custom_base.yaml" → 해당 경로 사용
    if isinstance(base_config_val, bool):
        base_path = "configs/base.yaml"
    else:
        base_path = base_config_val

    with open(base_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    return _deep_merge(base_cfg, exp_cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    """중첩 dict를 재귀적으로 병합합니다. override 값이 우선합니다."""
    merged = dict(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ── 실험 환경 정보 수집 ────────────────────────────────────────────────────────

def get_environment_info() -> dict:
    """OS, Python, GPU, 주요 패키지 버전 등 실험 환경 정보를 수집합니다."""
    env = {
        "os":             platform.system(),
        "os_version":     platform.version(),
        "python_version": sys.version.split()[0],
        "cpu_count":      os.cpu_count(),
    }

    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"]  = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
        else:
            env["gpu_name"]  = "N/A"
            env["gpu_count"] = 0
    except ImportError:
        env["cuda_available"] = False
        env["gpu_name"]       = "N/A"
        env["gpu_count"]      = 0

    for pkg in ("wandb", "chromadb", "openai", "langchain", "transformers"):
        try:
            import importlib.metadata
            env[f"{pkg}_version"] = importlib.metadata.version(pkg)
        except Exception:
            env[f"{pkg}_version"] = "unknown"

    return env


# ── Chroma DB 구축 & search_fn 생성 ───────────────────────────────────────────

def build_chroma_search_fn(chunks: list, cfg: dict):
    """
    Chroma DB에 chunk를 저장하고, 쿼리 → chunk_id 리스트를 반환하는 search_fn을 생성합니다.
    embedding provider에 따라 OpenAI / HuggingFace embedding function을 선택합니다.
    """
    import chromadb
    from chromadb.utils import embedding_functions

    provider = cfg["embedding"]["provider"]
    model    = cfg["embedding"]["model"]

    if provider == "openai":
        ef = embedding_functions.OpenAIEmbeddingFunction(
            model_name=model,
            api_key=cfg["embedding"].get("api_key"),  # None이면 환경변수 OPENAI_API_KEY 사용
        )
    elif provider == "huggingface":
        ef = embedding_functions.HuggingFaceEmbeddingFunction(model_name=model)
    else:
        raise ValueError(
            f"지원하지 않는 embedding provider: '{provider}'\n"
            f"사용 가능: openai | huggingface"
        )

    persist_dir = cfg["retrieval"]["persist_directory"]
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)

    collection_name = cfg["output"]["strategy_name"]
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    col = client.get_or_create_collection(collection_name, embedding_function=ef)

    BATCH = 500
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        col.add(
            ids       = [c.chunk_id for c in batch],
            documents = [c.text     for c in batch],
            metadatas = [c.metadata for c in batch],
        )

    top_k = cfg["retrieval"]["top_k"]

    def search_fn(query: str) -> list:
        results = col.query(query_texts=[query], n_results=top_k)
        return results["ids"][0]

    return search_fn


# ── Artifact 저장 헬퍼 ─────────────────────────────────────────────────────────

def _log_chunks_artifact(chunks: list, strategy_name: str, chunk_results_path: str) -> None:
    """chunks CSV를 W&B Artifact로 등록합니다."""
    path = Path(chunk_results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_id", "text", "strategy", "source"])
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
    print(f"  [chunks artifact] {path.name}  ({len(chunks)}개)")


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
    print(f"  [eval artifact] {Path(result_path).name}, {Path(summary_path).name}")


# ── LLM 호출 ──────────────────────────────────────────────────────────────────

def _call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    context: str,
    query: str,
    temperature: float,
    max_tokens: int,
    api_key: Optional[str] = None,
) -> str:
    """
    LLM을 호출해 응답 문자열을 반환합니다.
    provider: "openai" | "huggingface"
    """
    user_message = f"[컨텍스트]\n{context}\n\n[질문]\n{query}"

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model       = model,
            temperature = temperature,
            max_tokens  = max_tokens,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""

    elif provider == "huggingface":
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True)
        hf_model  = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config = quant_cfg,
            device_map          = "auto",
            trust_remote_code   = True,
        )

        messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_message}"}]
        prompt   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)

        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens = max_tokens,
                do_sample      = False,
            )

        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    else:
        raise ValueError(
            f"지원하지 않는 LLM provider: '{provider}'\n"
            f"사용 가능: openai | huggingface"
        )


# ── Generation 평가 ───────────────────────────────────────────────────────────

def _run_generation_eval(
    gt_df,
    search_fn,
    cfg: dict,
    run,
    step: int,
) -> dict:
    """
    LLM으로 각 쿼리에 대한 응답을 생성하고, 정량 지표를 계산해 W&B에 기록합니다.

    기록 항목:
      - query / ground_truth / generated_response      (per-query)
      - generation_time_s / answer_length              (per-query)
      - rouge1 / rouge2 / rougeL / exact_match         (per-query)
      - avg_* 집계 지표                                (summary)
      - generation/response_table                      (W&B Table)

    반환값:
      {
        "llm_model":             str,
        "avg_generation_time_s": float,
        "avg_answer_length":     float,
        "avg_rouge1":            float,
        "avg_rouge2":            float,
        "avg_rougeL":            float,
        "avg_exact_match":       float,
      }
    """
    gen_cfg = cfg.get("generation", {})
    if not gen_cfg:
        print("  [generation] config 없음 — generation 평가를 건너뜁니다.")
        return {}

    try:
        from rouge_score import rouge_scorer as rs_module
        rouge_scorer = rs_module.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )
    except ImportError:
        print("  ⚠ rouge_score 미설치 → pip install rouge-score")
        rouge_scorer = None

    llm_model     = gen_cfg.get("model", "unknown")
    provider      = gen_cfg.get("provider", "openai")
    temperature   = gen_cfg.get("temperature", 0.0)
    max_tokens    = gen_cfg.get("max_tokens", 512)
    system_prompt = gen_cfg.get(
        "system_prompt",
        "당신은 제안서 분석 전문가입니다. 주어진 문맥에만 철저히 기반하여 질문에 답하세요.\n"
        "문맥에 없는 내용이거나 확인 불가능한 정보라면 '문맥상 확인할 수 없습니다'라고 답하세요.",
    )

    print(f"\n  [Generation] LLM 응답 생성 중  model={llm_model}")

    rows             = []
    generation_times = []

    for _, row in gt_df.iterrows():
        query        = row["query"]
        ground_truth = row.get("ground_truth", "")

        retrieved_ids = search_fn(query)
        context       = "\n\n".join(retrieved_ids)

        t0 = time.time()
        generated_response = _call_llm(
            provider      = provider,
            model         = llm_model,
            system_prompt = system_prompt,
            context       = context,
            query         = query,
            temperature   = temperature,
            max_tokens    = max_tokens,
            api_key       = gen_cfg.get("api_key"),
        )
        generation_time = round(time.time() - t0, 3)
        generation_times.append(generation_time)

        # ROUGE 계산 (per-query)
        if rouge_scorer:
            scores  = rouge_scorer.score(ground_truth, generated_response)
            rouge1  = round(scores["rouge1"].fmeasure, 4)
            rouge2  = round(scores["rouge2"].fmeasure, 4)
            rougeL  = round(scores["rougeL"].fmeasure, 4)
        else:
            rouge1 = rouge2 = rougeL = None

        exact_match = int(ground_truth.strip() == generated_response.strip())

        rows.append({
            "query":              query,
            "ground_truth":       ground_truth,
            "generated_response": generated_response,
            "generation_time_s":  generation_time,
            "answer_length":      len(generated_response),
            "rouge1":             rouge1,
            "rouge2":             rouge2,
            "rougeL":             rougeL,
            "exact_match":        exact_match,
            "context_used":       context[:300] + "..." if len(context) > 300 else context,
        })

        print(
            f"    question_id={row.get('question_id', '?')}  "
            f"time={generation_time:.2f}s  "
            f"rouge1={rouge1 if rouge1 is not None else 'N/A'}"
        )

    # ── 집계 지표 ────────────────────────────────────────────────────────────
    n = max(len(rows), 1)

    avg_gen_time   = round(sum(generation_times) / n, 3)
    avg_ans_length = round(sum(r["answer_length"] for r in rows) / n, 1)

    if rouge_scorer:
        avg_rouge1      = round(sum(r["rouge1"]      for r in rows) / n, 4)
        avg_rouge2      = round(sum(r["rouge2"]      for r in rows) / n, 4)
        avg_rougeL      = round(sum(r["rougeL"]      for r in rows) / n, 4)
    else:
        avg_rouge1 = avg_rouge2 = avg_rougeL = None

    avg_exact_match = round(sum(r["exact_match"] for r in rows) / n, 4)

    # ── W&B Table ────────────────────────────────────────────────────────────
    import pandas as pd
    gen_df = pd.DataFrame(rows)

    table_cols = [
        "query", "ground_truth", "generated_response",
        "generation_time_s", "answer_length",
        "rouge1", "rouge2", "rougeL", "exact_match",
        "context_used",
    ]
    generation_table = wandb.Table(dataframe=gen_df[table_cols])

    log_dict = {
        "generation/llm_model":             llm_model,
        "generation/avg_generation_time_s": avg_gen_time,
        "generation/avg_answer_length":     avg_ans_length,
        "generation/avg_exact_match":       avg_exact_match,
        "generation/response_table":        generation_table,
    }
    if rouge_scorer:
        log_dict["generation/avg_rouge1"] = avg_rouge1
        log_dict["generation/avg_rouge2"] = avg_rouge2
        log_dict["generation/avg_rougeL"] = avg_rougeL

    wandb.log(log_dict, step=step)

    # run.summary에 최종 지표 기록 (대시보드 비교 컬럼)
    run.summary["generation/llm_model"]             = llm_model
    run.summary["generation/avg_generation_time_s"] = avg_gen_time
    run.summary["generation/avg_answer_length"]     = avg_ans_length
    run.summary["generation/avg_exact_match"]       = avg_exact_match
    if rouge_scorer:
        run.summary["generation/avg_rouge1"] = avg_rouge1
        run.summary["generation/avg_rouge2"] = avg_rouge2
        run.summary["generation/avg_rougeL"] = avg_rougeL

    print(
        f"  → generation 완료  "
        f"avg_time={avg_gen_time}s  "
        f"rouge1={avg_rouge1}  rouge2={avg_rouge2}  rougeL={avg_rougeL}  "
        f"exact={avg_exact_match}"
    )

    return {
        "llm_model":             llm_model,
        "avg_generation_time_s": avg_gen_time,
        "avg_answer_length":     avg_ans_length,
        "avg_rouge1":            avg_rouge1,
        "avg_rouge2":            avg_rouge2,
        "avg_rougeL":            avg_rougeL,
        "avg_exact_match":       avg_exact_match,
    }


# ── 단일 실험 실행 ────────────────────────────────────────────────────────────

def run_single_experiment(
    config_path: str,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str]  = None,
    experimenter: str = "",
) -> Optional[dict]:
    """
    config_path 하나에 대해 전체 파이프라인을 실행합니다.

    파이프라인 단계:
      [1] PDF → Markdown 파싱
      [2] 청킹 (fixed / markdown / semantic)
      [3] 임베딩 & Chroma DB 저장
      [4] Top-k 검색
      [5] Retrieval 평가  (recall, precision, MRR, nDCG)
      [6] Generation 평가 (ROUGE-1/2/L, Exact Match, generation_time)

    반환값 예시:
      {
        "strategy_name":          "baseline-markdown-bge-m3",
        "splitter":               "markdown",
        "chunk_size":             512,
        "chunk_overlap":          64,
        "embedding_model":        "BAAI/bge-m3",
        "llm_model":              "gpt-4o",
        "temperature":            0.2,
        "max_tokens":             1024,
        "context_recall":         0.75,
        "context_precision":      0.50,
        "mrr":                    0.83,
        "ndcg":                   0.80,
        "avg_rouge1":             0.42,
        "avg_rouge2":             0.21,
        "avg_rougeL":             0.38,
        "avg_exact_match":        0.10,
        "avg_generation_time_s":  1.23,
        "num_chunks":             142,
        "experimenter":           "홍길동",
        "environment":            "Linux / Python 3.11.0",
        "config_file":            "baseline.yaml",
        "status":                 "success",
      }
    """
    cfg           = load_config(config_path)
    strategy_name = cfg["output"]["strategy_name"]
    project_name  = Path(cfg["path"]["raw_pdf_file"]).stem

    _project      = wandb_project or cfg["wandb"]["project"]
    _entity       = wandb_entity  or cfg["wandb"]["entity"]
    _experimenter = experimenter  or cfg.get("wandb", {}).get("experimenter", "unknown")

    env_info = get_environment_info()

    # ── W&B 초기화 ───────────────────────────────────────────────────────────
    run = wandb.init(
        project  = _project,
        entity   = _entity,
        name     = strategy_name,
        config   = cfg,
        tags     = [
            cfg["preprocessing"]["splitter"],
            cfg["embedding"]["model"],
            cfg["embedding"]["provider"],
            cfg.get("generation", {}).get("model", "no-llm"),
        ],
        reinit   = "must",
    )

    # ── 메타 + 환경 정보 W&B 기록 ────────────────────────────────────────────
    wandb.log({
        # 모델 정보
        "meta/embedding_model":    cfg["embedding"]["model"],
        "meta/embedding_provider": cfg["embedding"]["provider"],
        "meta/llm_model":          cfg.get("generation", {}).get("model", "N/A"),
        "meta/llm_provider":       cfg.get("generation", {}).get("provider", "N/A"),
        # 청킹 전략 & 파라미터
        "meta/chunking_strategy":  cfg["preprocessing"]["splitter"],
        "meta/chunk_size":         cfg["preprocessing"].get("chunk_size", "N/A"),
        "meta/chunk_overlap":      cfg["preprocessing"].get("chunk_overlap", "N/A"),
        "meta/top_k":              cfg["retrieval"]["top_k"],
        "meta/temperature":        cfg.get("generation", {}).get("temperature", "N/A"),
        "meta/max_tokens":         cfg.get("generation", {}).get("max_tokens", "N/A"),
        # 실험 메타
        "meta/experimenter":       _experimenter,
        "meta/config_file":        Path(config_path).name,
        # 실험 환경
        "env/os":                  env_info["os"],
        "env/os_version":          env_info["os_version"],
        "env/python_version":      env_info["python_version"],
        "env/cpu_count":           env_info["cpu_count"],
        "env/cuda_available":      env_info["cuda_available"],
        "env/gpu_name":            env_info["gpu_name"],
        "env/gpu_count":           env_info["gpu_count"],
        "env/wandb_version":       env_info["wandb_version"],
        "env/chromadb_version":    env_info["chromadb_version"],
        "env/transformers_version":env_info["transformers_version"],
    }, step=0)

    run.config.update({"experimenter": _experimenter, "environment": env_info}, allow_val_change=True)

    print(f"\n{'='*60}")
    print(f"  실험 시작    : {strategy_name}")
    print(f"  config       : {config_path}")
    print(f"  splitter     = {cfg['preprocessing']['splitter']}")
    print(f"  chunk_size   = {cfg['preprocessing'].get('chunk_size', 'N/A')}")
    print(f"  chunk_overlap= {cfg['preprocessing'].get('chunk_overlap', 'N/A')}")
    print(f"  embedding    = {cfg['embedding']['model']}  ({cfg['embedding']['provider']})")
    print(f"  llm          = {cfg.get('generation', {}).get('model', 'N/A')}")
    print(f"  temperature  = {cfg.get('generation', {}).get('temperature', 'N/A')}")
    print(f"  max_tokens   = {cfg.get('generation', {}).get('max_tokens', 'N/A')}")
    print(f"  top_k        = {cfg['retrieval']['top_k']}")
    print(f"  experimenter = {_experimenter}")
    print(f"  env          = {env_info['os']} / Python {env_info['python_version']}")
    print(f"  gpu          = {env_info['gpu_name']}")
    print(f"  W&B run      : {run.url}")
    print(f"{'='*60}\n")

    result_row = {
        "strategy_name":         strategy_name,
        "splitter":              cfg["preprocessing"]["splitter"],
        "chunk_size":            cfg["preprocessing"].get("chunk_size", "N/A"),
        "chunk_overlap":         cfg["preprocessing"].get("chunk_overlap", "N/A"),
        "embedding_model":       cfg["embedding"]["model"],
        "llm_model":             cfg.get("generation", {}).get("model", "N/A"),
        "temperature":           cfg.get("generation", {}).get("temperature", "N/A"),
        "max_tokens":            cfg.get("generation", {}).get("max_tokens", "N/A"),
        "config_file":           Path(config_path).name,
        "experimenter":          _experimenter,
        "environment":           f"{env_info['os']} / Python {env_info['python_version']}",
        "num_chunks":            0,
        "avg_generation_time_s": None,
        "status":                "failed",
    }

    try:
        step = 1

        # ── [1] PDF → Markdown 파싱 ──────────────────────────────────────────
        print("[1/6] PDF 파싱 중")
        t0      = time.time()
        doc     = loader.extract_pdf(cfg["path"]["raw_pdf_file"])
        elapsed = round(time.time() - t0, 3)
        wandb.log({"pipeline/pages": doc.pages, "pipeline/load_time_s": elapsed}, step=step)
        step += 1
        print(f"  → {doc.pages}p  ({elapsed}s)")

        # ── [2] 청킹 ─────────────────────────────────────────────────────────
        print(f"\n[2/6] 청킹 수행 중  splitter={cfg['preprocessing']['splitter']}")
        t0      = time.time()
        chunks  = cleaner.get_chunks(doc.markdown, project_name, cfg)
        avg_len = sum(len(c.text) for c in chunks) / max(len(chunks), 1)
        elapsed = round(time.time() - t0, 3)
        wandb.log({
            "pipeline/num_chunks":      len(chunks),
            "pipeline/avg_chunk_len":   round(avg_len, 1),
            "pipeline/chunking_time_s": elapsed,
        }, step=step)
        step += 1
        print(f"  → {len(chunks)}개 청크  (평균 {avg_len:.0f}자,  {elapsed}s)")

        _log_chunks_artifact(chunks, strategy_name, cfg["output"]["chunk_results"])

        # ── [3] 임베딩 & Chroma DB 저장 ──────────────────────────────────────
        print("\n[3/6] 임베딩 생성 및 Chroma DB 저장 중")
        t0        = time.time()
        search_fn = build_chroma_search_fn(chunks, cfg)
        elapsed   = round(time.time() - t0, 3)
        wandb.log({"pipeline/embed_store_time_s": elapsed}, step=step)
        step += 1
        print(f"  → DB 저장 완료  ({elapsed}s)")

        # ── [4] Top-k 검색 ───────────────────────────────────────────────────
        print("\n[4/6] Top-k 검색 중")
        t0        = time.time()
        gt_df     = make_default_ground_truth_dataframe()
        result_df = evaluate_retrieval_dataframe(gt_df, search_fn, top_k=cfg["retrieval"]["top_k"])
        elapsed   = round(time.time() - t0, 3)
        wandb.log({"pipeline/search_time_s": elapsed}, step=step)
        step += 1

        # ── [5] Retrieval 평가 ────────────────────────────────────────────────
        print("\n[5/6] Retrieval 평가 중")
        summary = summarize_retrieval(result_df, strategy_name)

        per_query_table = wandb.Table(dataframe=result_df[[
            "query", "recall", "precision", "rr", "ndcg",
            "retrieved_ids", "relevant_ids",
        ]])
        wandb.log({
            "eval/context_recall":    summary.context_recall,
            "eval/context_precision": summary.context_precision,
            "eval/mrr":               summary.mrr,
            "eval/ndcg":              summary.ndcg,
            "eval/per_query_table":   per_query_table,
        }, step=step)
        step += 1

        run.summary["eval/context_recall"]    = summary.context_recall
        run.summary["eval/context_precision"] = summary.context_precision
        run.summary["eval/mrr"]               = summary.mrr
        run.summary["eval/ndcg"]              = summary.ndcg

        _log_eval_artifact(result_df, summary, cfg)

        print(
            f"\n  ✅ Retrieval  "
            f"recall={summary.context_recall:.4f}  "
            f"precision={summary.context_precision:.4f}  "
            f"MRR={summary.mrr:.4f}  "
            f"nDCG={summary.ndcg:.4f}"
        )

        # ── [6] Generation 평가 ───────────────────────────────────────────────
        print("\n[6/6] Generation 평가 중")
        gen_metrics = _run_generation_eval(
            gt_df     = gt_df,
            search_fn = search_fn,
            cfg       = cfg,
            run       = run,
            step      = step,
        )

        print(f"\n  W&B 대시보드: {run.url}")

        result_row.update({
            "context_recall":        summary.context_recall,
            "context_precision":     summary.context_precision,
            "mrr":                   summary.mrr,
            "ndcg":                  summary.ndcg,
            "num_chunks":            len(chunks),
            "avg_rouge1":            gen_metrics.get("avg_rouge1"),
            "avg_rouge2":            gen_metrics.get("avg_rouge2"),
            "avg_rougeL":            gen_metrics.get("avg_rougeL"),
            "avg_exact_match":       gen_metrics.get("avg_exact_match"),
            "avg_generation_time_s": gen_metrics.get("avg_generation_time_s"),
            "status":                "success",
        })
        return result_row

    except Exception:
        print(f"\n  ❌ 실험 실패: {strategy_name}")
        traceback.print_exc()
        return result_row

    finally:
        wandb.finish()


# ── 전체 실험 완료 후 비교 테이블 W&B 기록 ────────────────────────────────────

def log_comparison_table(all_results: list, cfg: dict) -> None:
    """
    모든 실험 결과를 하나의 W&B run에 비교 테이블로 기록합니다.
    대시보드에서 실험 간 성능을 한눈에 비교하는 용도입니다.
    """
    import pandas as pd

    if not any(r["status"] == "success" for r in all_results):
        print("\n  성공한 실험이 없어 비교 테이블을 생성하지 않습니다.")
        return

    run = wandb.init(
        project  = cfg["wandb"]["project"],
        entity   = cfg["wandb"]["entity"],
        name     = "experiment_comparison",
        job_type = "comparison",
        reinit   = "must",
    )

    success_results = [r for r in all_results if r["status"] == "success"]
    df              = pd.DataFrame(success_results)

    # ── 지표별 최고 실험 하이라이트 ─────────────────────────────────────────
    def _best(col):
        return df.loc[df[col].idxmax(), "strategy_name"] if col in df.columns else "N/A"

    run.summary["best/recall_strategy"]      = _best("context_recall")
    run.summary["best/precision_strategy"]   = _best("context_precision")
    run.summary["best/mrr_strategy"]         = _best("mrr")
    run.summary["best/ndcg_strategy"]        = _best("ndcg")
    run.summary["best/rouge1_strategy"]      = _best("avg_rouge1")
    run.summary["best/rougeL_strategy"]      = _best("avg_rougeL")
    run.summary["best/exact_match_strategy"] = _best("avg_exact_match")

    comparison_cols = [
        "strategy_name", "splitter", "chunk_size", "chunk_overlap",
        "embedding_model", "llm_model", "temperature", "max_tokens",
        # retrieval 지표
        "context_recall", "context_precision", "mrr", "ndcg",
        # generation 지표
        "avg_rouge1", "avg_rouge2", "avg_rougeL",
        "avg_exact_match", "avg_generation_time_s",
        # 메타
        "num_chunks", "experimenter", "environment", "config_file", "status",
    ]
    comparison_cols   = [c for c in comparison_cols if c in df.columns]
    comparison_table  = wandb.Table(dataframe=df[comparison_cols])
    wandb.log({"comparison/all_experiments": comparison_table})
    wandb.finish()

    # ── 콘솔 요약 출력 ───────────────────────────────────────────────────────
    print_cols = [c for c in [
        "strategy_name",
        "context_recall", "context_precision", "mrr", "ndcg",
        "avg_rouge1", "avg_rougeL", "avg_exact_match",
        "avg_generation_time_s",
    ] if c in df.columns]

    print(f"\n{'='*70}")
    print("  전체 실험 결과 비교")
    print(f"{'='*70}")
    print(df[print_cols].to_string(index=False))
    print(f"\n  🏆 Best recall      : {run.summary.get('best/recall_strategy')}")
    print(f"  🏆 Best precision   : {run.summary.get('best/precision_strategy')}")
    print(f"  🏆 Best MRR         : {run.summary.get('best/mrr_strategy')}")
    print(f"  🏆 Best nDCG        : {run.summary.get('best/ndcg_strategy')}")
    print(f"  🏆 Best ROUGE-1     : {run.summary.get('best/rouge1_strategy')}")
    print(f"  🏆 Best ROUGE-L     : {run.summary.get('best/rougeL_strategy')}")
    print(f"  🏆 Best Exact Match : {run.summary.get('best/exact_match_strategy')}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG 실험 일괄 실행 스크립트")

    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--configs",
        nargs    = "+",
        metavar  = "CONFIG_YAML",
        help     = "실험 config yaml 경로를 하나 이상 지정합니다.",
    )
    config_group.add_argument(
        "--config_dir",
        metavar  = "DIR",
        help     = "폴더 안의 모든 *.yaml을 자동으로 탐색합니다.",
    )

    parser.add_argument("--wandb_project",  default=None, help="W&B 프로젝트명")
    parser.add_argument("--wandb_entity",   default=None, help="W&B 엔티티명")
    parser.add_argument("--experimenter",   default="",   help="실험 진행자 이름")
    parser.add_argument(
        "--no_skip_failed",
        action = "store_true",
        help   = "개별 실험 실패 시 전체를 즉시 중단합니다 (기본값: 실패해도 다음 실험 계속)",
    )
    parser.add_argument(
        "--dry_run",
        action = "store_true",
        help   = "config 파일 목록만 출력하고 실제 실험은 실행하지 않습니다",
    )

    args = parser.parse_args()

    if args.configs:
        config_paths = [Path(p) for p in args.configs]
    else:
        config_paths = sorted(Path(args.config_dir).glob("*.yaml"))

    if not config_paths:
        print("  ❌ 실험 config 파일을 찾지 못했습니다.")
        return

    print(f"\n  실행 예정 실험 수: {len(config_paths)}개")
    for i, p in enumerate(config_paths, 1):
        print(f"    [{i:02d}] {p}")

    if args.dry_run:
        print("\n  --dry_run 모드: 실제 실험을 실행하지 않습니다.")
        return

    wandb.login()

    all_results = []
    total       = len(config_paths)

    for idx, config_path in enumerate(config_paths, 1):
        print(f"\n\n{'#'*60}")
        print(f"  [{idx}/{total}] 실험 실행: {config_path.name}")
        print(f"{'#'*60}")

        result = run_single_experiment(
            config_path   = str(config_path),
            wandb_project = args.wandb_project,
            wandb_entity  = args.wandb_entity,
            experimenter  = args.experimenter,
        )

        if result is not None:
            all_results.append(result)

        if result and result["status"] == "failed" and args.no_skip_failed:
            print("\n  --no_skip_failed: 실험 중단")
            break

    if len(all_results) > 1:
        with open("configs/base.yaml", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        if args.wandb_project:
            base_cfg["wandb"]["project"] = args.wandb_project
        if args.wandb_entity:
            base_cfg["wandb"]["entity"]  = args.wandb_entity
        log_comparison_table(all_results, base_cfg)

    success = sum(1 for r in all_results if r["status"] == "success")
    failed  = sum(1 for r in all_results if r["status"] == "failed")
    print(f"\n\n{'='*60}")
    print(f"  실험 완료  성공: {success}개  실패: {failed}개  (총 {total}개)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
