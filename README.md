# 입찰메이트 RAG System

복잡한 기업 및 정부 제안요청서(RFP)를 분석하고  
핵심 정보를 검색·요약·질의응답할 수 있는  
RAG(Retrieval-Augmented Generation) 기반 문서 분석 시스템 구축 프로젝트

---

# 프로젝트 소개

본 프로젝트는 자연어처리(NLP) 및 LLM 기술을 활용하여  
다양한 형태의 기업 및 정부 제안요청서(RFP)를 효과적으로 이해하고,  
사용자의 질문에 맞는 정보를 제공할 수 있는  
RAG 기반 문서 분석 시스템을 구축하는 것을 목표로 합니다.

---

# 프로젝트 배경

공공입찰 컨설팅 스타트업 "입찰메이트"를 가상의 서비스 환경으로 설정하였습니다.

공공기관 및 기업의 제안요청서(RFP)는 문서 길이가 길고 구조가 복잡하여  
컨설턴트들이 모든 문서를 직접 읽고 핵심 정보를 파악하는 데 많은 시간이 소요됩니다.

특히 다음과 같은 정보들은 빠르게 확인되어야 합니다.

- 사업 목적
- 예산 규모
- 과업 범위
- 제출 방식
- 평가 기준
- 입찰 참가 조건
- 수행 일정

본 프로젝트에서는 이러한 문제를 해결하기 위해  
RAG 기반 문서 이해 시스템을 구축하고자 합니다.

---

# 프로젝트 목표

- PDF 기반 RFP 문서 처리 파이프라인 구축
- 문서 검색(Retrieval) 기반 질의응답 시스템 구현
- 핵심 정보 추출 및 요약 기능 구현
- 다양한 Embedding / Retrieval 전략 비교 실험
- Hallucination을 최소화한 근거 기반 응답 생성
- 성능 평가 지표 설계 및 분석

---

# 🛠 Communication & Collaboration Tools

| Tool | Description | Link |
|---|---|---|
| Notion | 프로젝트 문서 및 일정 관리 | https://www.notion.so/b432b0aa2cb2828ba6bd016c0ba6e0e1 |
| Figma | 시스템 구조 및 아이디어 협업 | https://www.figma.com/board/R6X9kT6zLQerI6pxw5P3ia/1%ED%8C%80?node-id=0-1&p=f&t=gW77nPC7SN8GSwhr-0 |
| Weights & Biases (W&B) | 실험 로그 및 성능 추적 | https://wandb.ai/csd1345- |
| DVC | 데이터 및 모델 버전 관리 | 추가 예정 |

---

# 팀원 별 역활

|팀원 이름|역활|
|---------|-----|
|김경제|Evaluation 담당 (팀장)|
|김영성|Data 담당|
|신희정|Retrieval 담당|
|엄지영|Retrieval 담당|
|황예원|Generation 담당|

PM(Project Manager) : 일마다 돌아가면서 담당함.

---

# 📂 데이터셋

본 프로젝트에서는 실제 기업 및 공공기관 제안요청서(RFP) 문서를 활용합니다.

## 제공 데이터

- RFP 문서 100개
- PDF/HWP 형식 (PDF : 4개 / HWP : 96개)
- 문서별 메타데이터 포함

## 메타데이터 정보

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| 공고 번호 | str | 입찰 공고 식별자 |
| 공고 차수 | float64 | 공고 회차 |
| 사업명 | str | RFP 사업명 |
| 사업 금액 | float64 | 예산 (원 단위) |
| 발주 기관 | str | 발주처 |
| 공개 일자 | str | 공고 공개 일시 |
| 입찰 참여 시작일 | str | 입찰 시작 일시 |
| 입찰 참여 마감일 | str | 입찰 마감 일시 |
| 사업 요약 | str | 핵심 내용 요약 (bullet) |
| 파일형식 | str | hwp / pdf |
| 파일명 | str | 원본 파일명 |
| 텍스트 | str | 파일에서 추출된 텍스트 |

---

# 사용 모델 (시나리오 A : GCP 실행 기반)

| |모델 이름|HuggingFace URL|
|---|---------|--------------|
|임베딩 모델|BAAI/bge-m3|https://huggingface.co/BAAI/bge-m3|
|LLM 모델|Qwen/Qwen3-8B|https://huggingface.co/Qwen/Qwen3-8B|

---

# 사용 모델 (시나리오 B : LLM API 기반)

| |모델 이름|OpenAI Developers URL|비용 (100만 토큰 당)|
|---|---------|---------|-----|
|임베딩 모델|OpenAI/text-embedding-3-small|https://developers.openai.com/api/docs/models/text-embedding-3-small|0.02$|
|LLM 모델|OpenAI/GPT-5-mini|https://developers.openai.com/api/docs/models/gpt-5-mini|입력 : 0.25$, 출력 : 2$|

---

# Vector DB

두 시나리오 모두 ChromaDB 사용 (https://www.trychroma.com/products/chromadb)

---

# 개인 협업일지

|팀원 이름|협업일지 URL|
|---------|------------|
|김경제|https://shrub-weight-c16.notion.site/AI_9-_-_Daily_-_part_2-35f2b0aa2cb28026af57ee87c877ea57?source=copy_link|
|김영성|https://velog.io/@csd1345/2026-05-12-협업일지|
|신희정|https://app.notion.com/p/AI09-35fbb18aad6580fba225fde3af002c91?source=copy_link|
|엄지영|https://www.notion.so/Daily-1-3609cec3d60f80f2b057ffd07ced50e3?source=copy_link|
|황예원|https://www.notion.so/26-5-12-26-6-4-35f16104cddd80b7b8dbdb3faa256718?source=copy_link|
