# src/evaluation

이 디렉터리는 RAG 파이프라인의 검색 품질과 생성 답변 품질을 평가하는 공통 모듈을 담고 있습니다. 실험 스크립트에서 반복해서 쓰는 지표 계산 로직을 이곳에 모아 결과 비교가 일관되도록 합니다.

## 주요 파일

`ground_truth.py`는 평가 질문과 정답 데이터를 문서별로 관리합니다. `GROUND_TRUTH_REGISTRY`에 문서 ID별 Ground Truth가 등록되어 있으며, `make_ground_truth_dataframe` 함수가 실험에서 사용할 DataFrame을 생성합니다.

`retrieval.py`는 검색 결과 평가 지표를 계산합니다. 주요 지표는 context recall, context precision, MRR, nDCG입니다. 검색 결과가 문자열, 리스트, JSON 형태로 들어와도 `parse_retrieved_contexts`를 통해 평가 가능한 형태로 변환합니다.

`generation.py`는 생성 답변 평가 지표를 계산합니다. BLEU, ROUGE, token F1 같은 텍스트 유사도 지표와 함께 faithfulness, answer relevance, metadata match 점수를 제공합니다. `judge_fn`을 넘기면 LLM judge 기반 평가도 함께 사용할 수 있습니다.

`__init__.py`는 외부에서 자주 쓰는 평가 함수들을 한 번에 import할 수 있도록 공개 인터페이스를 정리합니다.

## Retrieval 평가 흐름

실험 스크립트는 Ground Truth 질문별로 검색된 청크를 `retrieved_contexts`에 담고, `evaluate_retrieval_dataframe`을 호출합니다. 이 함수는 각 행에 대해 검색된 문맥과 정답을 비교해 세부 지표를 붙입니다. 이후 `summarize_retrieval`이 전략별 평균 요약을 만듭니다.

## Generation 평가 흐름

생성 평가에서는 질문, 정답, 검색 문맥, 모델 답변이 필요합니다. `evaluate_generation_dataframe`은 행 단위로 BLEU, ROUGE-1, ROUGE-2, ROUGE-L, token precision/recall/F1, faithfulness, answer relevance, metadata match를 계산합니다. 마지막으로 `summarize_by_strategy`를 사용해 실험 전략별 평균 점수를 확인할 수 있습니다.

## CLI 사용

`retrieval.py`와 `generation.py`는 단독 실행용 CLI도 포함합니다. CSV 또는 Excel 형태의 평가 입력 파일을 넣고 `--output` 경로를 지정하면 평가 결과 CSV를 저장할 수 있습니다.

## 수정 시 주의사항

- 새로운 평가 지표를 추가할 때는 행 단위 함수와 DataFrame 단위 함수가 모두 같은 컬럼 규칙을 따르도록 맞추세요.
- Ground Truth를 추가할 때는 `document_id`가 `configs/base.yaml`의 문서 설정과 일치해야 합니다.
- LLM judge는 API 키와 외부 모델 호출 비용이 필요하므로, 휴리스틱 지표만으로도 실행 가능한 상태를 유지하는 것이 좋습니다.
