# experiments

이 디렉터리는 실제 RAG 실험을 실행하는 스크립트를 담고 있습니다. 설정 파일을 읽어 PDF 전처리, 청킹, 임베딩, 벡터 DB 저장, 검색 평가, 답변 생성 평가까지 이어지는 baseline 파이프라인을 수행합니다.

## 실행 순서

1. `baseline_embedding.py`를 실행해 PDF를 Markdown으로 추출하고 청크를 만든 뒤 Chroma DB를 저장합니다.
2. `baseline_generation.py`를 실행해 저장된 Chroma DB를 로드하고, Ground Truth 문항별 검색 결과를 바탕으로 답변을 생성합니다.
3. 각 단계에서 생성된 상세 결과와 요약 결과는 `outputs/` 아래 CSV로 저장됩니다.

`outputs/`는 `.gitignore`에 포함되어 있으므로 실험 산출물은 Git에 커밋되지 않습니다.

## 파일별 역할

`baseline_embedding.py`는 검색 인덱스를 만드는 단계입니다. 주요 흐름은 설정 병합, PDF 경로 해석, PDF Markdown 추출, Markdown 청킹, 청크 메타데이터 생성, Chroma DB 저장, retrieval 평가 결과 저장입니다. 이 스크립트가 만들어 둔 `persist_directory`가 generation 단계의 입력이 됩니다.

`baseline_generation.py`는 저장된 검색 인덱스를 활용해 답변 생성과 generation 평가를 수행합니다. embedding 단계와 동일한 임베딩 모델로 Chroma DB를 로드하고, Ground Truth 질문마다 관련 청크를 검색한 뒤 생성 모델에 문맥과 질문을 전달합니다. 이후 공통 평가 모듈을 호출해 BLEU, ROUGE, token F1, faithfulness, answer relevance, metadata match 점수를 계산합니다.

## 설정 파일

기본 설정 파일은 다음 경로를 사용합니다.

```bash
configs/experiments/bge-m3_qwen3-8B.yaml
```

다른 실험을 추가할 때는 `configs/experiments/` 아래에 새 YAML 파일을 만들고 `--config` 인자로 넘기면 됩니다.

## 실행 예시

```bash
python experiments/baseline_embedding.py --config configs/experiments/bge-m3_qwen3-8B.yaml
python experiments/baseline_generation.py --config configs/experiments/bge-m3_qwen3-8B.yaml
```

Hugging Face 모델을 CUDA에서 실행하도록 설정되어 있으므로 GPU 환경을 권장합니다. OpenAI provider나 judge 모델을 사용할 경우 `.env` 또는 환경 변수에 `OPENAI_API_KEY`가 필요합니다.
