# configs/experiments

이 디렉터리는 개별 실험별 YAML 설정을 보관합니다. 공통 설정은 `../base.yaml`에 두고, 여기서는 모델, 청킹 파라미터, 검색 개수, 저장 경로처럼 실험마다 달라지는 값만 명시합니다.

## 현재 실험 파일

- `bge-m3_qwen3-8B.yaml`: `BAAI/bge-m3` 임베딩 모델과 `Qwen/Qwen3-8B` 생성 모델을 사용하는 baseline RAG 실험 설정입니다.

## 주요 설정 항목

`base_config`는 병합할 공통 설정 파일 경로입니다. 대부분의 실험은 `configs/base.yaml`을 기준으로 시작합니다.

`preprocessing`은 PDF에서 추출한 Markdown 텍스트를 청크로 나누는 방법을 정합니다. 현재 baseline은 `markdown` splitter를 사용하며, `chunk_size`, `chunk_overlap`, `min_chunk_len` 값으로 청크 길이와 겹침 범위를 조정합니다.

`embedding`은 검색 인덱스용 벡터를 생성하는 모델 설정입니다. `provider`가 `huggingface`이면 `langchain-huggingface` 기반 임베딩을 사용하고, `openai`이면 OpenAI 임베딩을 사용할 수 있도록 스크립트가 분기합니다.

`retrieval`은 Chroma 벡터 DB와 검색 설정을 관리합니다. `persist_directory`는 embedding 단계에서 만든 DB가 저장되는 위치이며, generation 단계에서는 같은 경로를 다시 읽습니다. 따라서 이 값은 두 단계가 반드시 공유해야 합니다.

`generation`은 답변 생성 모델과 평가용 judge 모델을 정의합니다. `judge_model`이 설정되어 있고 `OPENAI_API_KEY`가 있으면 생성 평가에서 LLM judge 점수를 함께 계산합니다.

`output`은 청킹 결과, retrieval 평가 결과, generation 평가 결과가 저장될 CSV 경로를 정의합니다. 같은 실험은 하나의 `strategy_name`과 파일명 slug를 공유하도록 맞추면 결과 추적이 쉬워집니다.

## 실행 예시

```bash
python experiments/baseline_embedding.py --config configs/experiments/bge-m3_qwen3-8B.yaml
python experiments/baseline_generation.py --config configs/experiments/bge-m3_qwen3-8B.yaml
```

먼저 embedding 스크립트를 실행해 Chroma DB를 만든 뒤 generation 스크립트를 실행해야 합니다.
