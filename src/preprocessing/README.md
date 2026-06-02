# src/preprocessing

이 디렉터리는 원본 RFP PDF를 RAG 검색에 사용할 수 있는 텍스트 청크로 변환하는 전처리 모듈을 담고 있습니다.

## 주요 파일

`loader.py`는 PDF 파일을 Markdown 텍스트로 추출합니다. 내부적으로 `pymupdf4llm.to_markdown`을 사용하며, 필요하면 PDF 페이지 범위와 이미지 저장 옵션을 설정할 수 있습니다. 파일이 존재하지 않을 경우 `FileNotFoundError`를 발생시켜 잘못된 데이터 경로를 빠르게 확인할 수 있게 합니다.

`cleaner.py`는 추출된 텍스트를 정리하고 청크로 분할합니다. `RFPTextCleaner` 클래스는 설정 파일의 `preprocessing.chunk_size`, `chunk_overlap`, `min_chunk_len` 값을 사용합니다.

`PyMuPDFLLM_baseline.ipynb`는 PyMuPDF 기반 PDF 추출을 실험하던 노트북입니다. 현재 실험 스크립트의 정식 실행 경로는 `.py` 모듈과 `experiments/` 스크립트입니다.

## 청킹 방식

`run_markdown_chunking`은 Markdown heading을 기준으로 섹션을 나누고, 긴 섹션은 설정된 chunk size와 overlap에 맞춰 추가 분할합니다. 각 청크 앞에는 문서명과 섹션 제목을 붙여 검색 결과에서 출처 맥락을 유지합니다.

`run_fixed_size_chunking`은 전체 텍스트를 고정 길이로 자릅니다. 구조 정보가 약한 문서에 사용할 수 있지만, 현재 baseline 실험은 Markdown 기반 청킹을 사용합니다.

`run_semantic_chunking`은 LangChain의 `RecursiveCharacterTextSplitter`를 사용해 줄바꿈, 번호, 한글 목차 기호, 공백 등의 separator 우선순위를 고려해 텍스트를 나눕니다.

## 입력과 출력

입력은 `configs/base.yaml`의 `path.file_dir` 아래 PDF 파일입니다. 출력은 문자열 청크 리스트이며, 실험 스크립트에서 LangChain `Document` 객체로 감싼 뒤 Chroma DB에 저장합니다.

## 수정 시 주의사항

- 청크 앞에 붙는 문서명과 섹션 정보는 검색 결과 해석과 평가에 도움이 되므로 제거하지 않는 편이 좋습니다.
- `chunk_size`와 `chunk_overlap`을 바꾸면 retrieval/generation 결과 파일명과 `strategy_name`도 함께 맞추는 것이 좋습니다.
- 이미지 추출을 켜면 `outputs/images` 아래에 파일이 생기며, 해당 경로는 Git에 포함되지 않습니다.
