# src

이 디렉터리는 RFP 분석용 RAG 파이프라인에서 재사용되는 핵심 모듈을 모아 둔 곳입니다. 실험 스크립트는 이 모듈들을 불러와 PDF 전처리, 청킹, Ground Truth 생성, retrieval 평가, generation 평가를 수행합니다.

## 디렉터리 구조

- `preprocessing/`: PDF를 Markdown 텍스트로 추출하고 검색에 적합한 청크로 나누는 전처리 모듈입니다.
- `evaluation/`: 검색 결과와 생성 답변을 평가하는 공통 지표 함수와 Ground Truth 데이터를 관리합니다.

## 현재 파이프라인에서의 역할

`experiments/baseline_embedding.py`는 `src.preprocessing.loader.extract_pdf`와 `src.preprocessing.cleaner.RFPTextCleaner`를 사용해 PDF를 청크로 변환합니다. 이후 `src.evaluation.retrieval`의 평가 함수를 호출해 검색 결과 품질을 계산합니다.

`experiments/baseline_generation.py`는 `src.evaluation.ground_truth`에서 평가 문항을 가져오고, `src.evaluation.generation`의 공통 평가 함수를 사용해 생성 답변 품질을 계산합니다.

## 개발 시 참고사항

- 실험 스크립트에 직접 평가 로직을 추가하기보다 `evaluation/` 아래에 공통 함수로 분리하는 것을 권장합니다.
- PDF 추출이나 청킹 방식 변경은 `preprocessing/`에서 먼저 처리하면 실험 스크립트 변경을 줄일 수 있습니다.
