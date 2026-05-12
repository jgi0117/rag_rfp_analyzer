"""
RFP RAG 파이프라인 - 핵심 모듈
- 문서 파싱 및 청킹
- 임베딩 생성
- ChromaDB 벡터 저장소 구축
- 메타데이터 필터링
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────
# 1. 데이터 모델
# ─────────────────────────────────────────

@dataclass
class RFPMetadata:
    """RFP 문서 메타데이터 스키마"""
    doc_id: str
    title: str
    issuing_org: str          # 발주 기관
    org_type: str             # government / enterprise / public
    budget_min: float         # 예산 하한 (만원)
    budget_max: float         # 예산 상한 (만원)
    deadline: str             # 제출 마감일 (YYYY-MM-DD)
    domain: str               # IT / 건설 / 컨설팅 / 연구개발 등
    region: str               # 서울 / 경기 / 전국 등
    submission_method: str    # 온라인 / 방문 / 우편
    source_file: str
    ingested_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_chroma_meta(self) -> dict:
        """ChromaDB에 저장 가능한 flat dict로 변환 (float/int/str/bool만 허용)"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "issuing_org": self.issuing_org,
            "org_type": self.org_type,
            "budget_min": float(self.budget_min),
            "budget_max": float(self.budget_max),
            "deadline": self.deadline,
            "domain": self.domain,
            "region": self.region,
            "submission_method": self.submission_method,
            "source_file": self.source_file,
            "ingested_at": self.ingested_at,
        }


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str       # 청크가 속한 섹션 헤더
    chunk_index: int
    metadata: RFPMetadata


# ─────────────────────────────────────────
# 2. PDF 파싱 & 섹션 분리
# ─────────────────────────────────────────

class RFPParser:
    """
    PDF/텍스트 RFP 문서를 파싱하여 섹션 단위로 분리합니다.
    섹션 헤더 패턴을 정규식으로 탐지합니다.
    """

    SECTION_PATTERNS = [
        r"^제\s*\d+\s*조",          # 제1조, 제 2 조
        r"^\d+\.\s+[가-힣A-Za-z]",  # 1. 사업개요
        r"^[IVX]+\.\s+[가-힣A-Za-z]", # I. 사업목적
        r"^[가-힣]{2,10}\s*$",       # 사업개요 (단독 행)
    ]

    def __init__(self):
        self.section_re = re.compile(
            "|".join(f"({p})" for p in self.SECTION_PATTERNS),
            re.MULTILINE
        )

    def parse_pdf(self, filepath: str) -> dict[str, str]:
        """PDF → {섹션명: 텍스트} 딕셔너리 반환"""
        doc = fitz.open(filepath)
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return self._split_sections(full_text)

    def parse_text(self, text: str) -> dict[str, str]:
        return self._split_sections(text)

    def _split_sections(self, text: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        lines = text.split("\n")
        current_section = "전문"
        buffer: list[str] = []

        for line in lines:
            stripped = line.strip()
            if self.section_re.match(stripped) and len(stripped) < 60:
                if buffer:
                    sections[current_section] = "\n".join(buffer).strip()
                current_section = stripped
                buffer = []
            else:
                buffer.append(line)

        if buffer:
            sections[current_section] = "\n".join(buffer).strip()

        return sections


# ─────────────────────────────────────────
# 3. 청킹 전략
# ─────────────────────────────────────────

class RFPChunker:
    """
    섹션별로 청킹하되, 섹션 컨텍스트를 prefix로 보존합니다.
    Sliding window로 청크 간 overlap을 유지합니다.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", " "],
            length_function=len,
        )

    def chunk_sections(
        self,
        sections: dict[str, str],
        metadata: RFPMetadata,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        idx = 0

        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue

            # 섹션 헤더를 각 청크 prefix로 붙여 컨텍스트 보존
            prefixed = f"[섹션: {section_name}]\n{section_text}"
            raw_chunks = self.splitter.split_text(prefixed)

            for chunk_text in raw_chunks:
                chunk_id = hashlib.md5(
                    f"{metadata.doc_id}_{idx}".encode()
                ).hexdigest()[:12]

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=metadata.doc_id,
                    text=chunk_text,
                    section=section_name,
                    chunk_index=idx,
                    metadata=metadata,
                ))
                idx += 1

        return chunks


# ─────────────────────────────────────────
# 4. 임베딩 모델
# ─────────────────────────────────────────

class EmbeddingModel:
    """
    한국어 특화 임베딩 모델을 사용합니다.
    - 기본: jhgan/ko-sroberta-multitask (한국어 SRoBERTa)
    - 대안: snunlp/KR-ELECTRA-discriminator (전기문 특화)
    - 대안: BAAI/bge-m3 (다국어, 고성능)
    """

    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        print(f"[EmbeddingModel] 모델 로딩: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """텍스트 리스트 → 임베딩 벡터 리스트"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # 코사인 유사도 최적화
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """단일 쿼리 임베딩 (검색 시 사용)"""
        vec = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()


# ─────────────────────────────────────────
# 5. 벡터 DB (ChromaDB)
# ─────────────────────────────────────────

class RFPVectorStore:
    """
    ChromaDB 기반 벡터 저장소.
    - 컬렉션: rfp_chunks
    - 메타데이터 필터링 지원
    - 배치 upsert 지원
    """

    COLLECTION_NAME = "rfp_chunks"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # 코사인 유사도 사용
        )
        print(f"[VectorStore] 컬렉션 '{self.COLLECTION_NAME}' 준비 완료")

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """청크와 임베딩을 벡터 DB에 저장 (중복 시 업데이트)"""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = []

        for c in chunks:
            meta = c.metadata.to_chroma_meta()
            meta["section"] = c.section
            meta["chunk_index"] = c.chunk_index
            metadatas.append(meta)

        # ChromaDB 배치 제한(5461) 대응
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )

        print(f"[VectorStore] {len(ids)}개 청크 저장 완료")

    def delete_document(self, doc_id: str) -> None:
        """특정 문서의 모든 청크 삭제"""
        self.collection.delete(where={"doc_id": {"$eq": doc_id}})
        print(f"[VectorStore] 문서 '{doc_id}' 삭제 완료")

    def get_stats(self) -> dict:
        return {
            "total_chunks": self.collection.count(),
            "collection": self.COLLECTION_NAME,
        }


# ─────────────────────────────────────────
# 6. 인제스천 파이프라인 (통합)
# ─────────────────────────────────────────

class RFPIngestionPipeline:
    """
    RFP 문서 → 파싱 → 청킹 → 임베딩 → 벡터 DB 저장까지
    전체 인제스천 파이프라인을 관리합니다.
    """

    def __init__(
        self,
        embed_model_name: str = "jhgan/ko-sroberta-multitask",
        persist_dir: str = "./chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.parser = RFPParser()
        self.chunker = RFPChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingModel(embed_model_name)
        self.vector_store = RFPVectorStore(persist_dir)

    def ingest_pdf(self, filepath: str, metadata: RFPMetadata) -> int:
        """PDF 파일 인제스천 → 저장된 청크 수 반환"""
        print(f"\n[Ingest] 파일 처리 시작: {filepath}")

        # 1) 파싱
        sections = self.parser.parse_pdf(filepath)
        print(f"  → {len(sections)}개 섹션 파싱 완료")

        # 2) 청킹
        chunks = self.chunker.chunk_sections(sections, metadata)
        print(f"  → {len(chunks)}개 청크 생성")

        # 3) 임베딩
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)

        # 4) 저장
        self.vector_store.upsert_chunks(chunks, embeddings)
        return len(chunks)

    def ingest_text(self, text: str, metadata: RFPMetadata) -> int:
        """텍스트 직접 인제스천 (테스트/API용)"""
        sections = self.parser.parse_text(text)
        chunks = self.chunker.chunk_sections(sections, metadata)
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)
        self.vector_store.upsert_chunks(chunks, embeddings)
        return len(chunks)


# ─────────────────────────────────────────
# 7. 사용 예시
# ─────────────────────────────────────────

if __name__ == "__main__":
    pipeline = RFPIngestionPipeline(
        embed_model_name="jhgan/ko-sroberta-multitask",
        persist_dir="./chroma_db",
        chunk_size=512,
        chunk_overlap=64,
    )

    # 예시 메타데이터 (실제 RFP 파싱 후 추출하거나 수동 입력)
    meta = RFPMetadata(
        doc_id="rfp_2024_001",
        title="2024년 공공기관 디지털 전환 컨설팅 용역",
        issuing_org="한국정보화진흥원",
        org_type="government",
        budget_min=50000,
        budget_max=100000,
        deadline="2024-03-31",
        domain="IT",
        region="서울",
        submission_method="온라인",
        source_file="rfp_2024_001.pdf",
    )

    # PDF 인제스천 (파일이 있는 경우)
    # n = pipeline.ingest_pdf("rfp_2024_001.pdf", meta)

    # 텍스트 인제스천 (테스트용)
    sample_text = """
    제1조 사업개요
    본 사업은 공공기관의 디지털 전환을 위한 컨설팅 용역으로,
    클라우드 전환 전략 수립 및 데이터 거버넌스 체계 구축을 목표로 합니다.

    제2조 예산 및 기간
    총 사업 예산은 8억원이며, 사업 기간은 2024년 4월부터 12월까지입니다.

    제3조 제출 방식
    제안서는 온라인 시스템(나라장터)을 통해 제출하며,
    마감일은 2024년 3월 31일 오후 5시입니다.

    제4조 주요 요구사항
    - 클라우드 전환 로드맵 수립
    - 데이터 거버넌스 프레임워크 설계
    - 보안 아키텍처 검토 및 개선안 제시
    - 직원 교육 프로그램 개발
    """

    n = pipeline.ingest_text(sample_text, meta)
    print(f"\n총 {n}개 청크 인제스천 완료")
    print(f"DB 현황: {pipeline.vector_store.get_stats()}")




    # ──────────────────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────────────
    # retrieval
    # ──────────────────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────────────



    """
RFP 고도화 Retrieval 모듈
- Hybrid Search: Dense + Sparse (BM25)
- Metadata Filtering: 예산/기관/기간/도메인 복합 필터
- Reranking: Cross-Encoder 재순위화
- MMR: 다양성 보장 검색
- Query Expansion: 동의어/유사 표현 확장
"""

import re
from typing import Optional, Any
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.config import Settings

from rfp_rag_pipeline import EmbeddingModel


# ─────────────────────────────────────────
# 1. 결과 데이터 모델
# ─────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    text: str
    section: str
    score: float                   # 최종 점수
    dense_score: float             # 벡터 유사도 점수
    bm25_score: float              # BM25 점수
    rerank_score: Optional[float]  # Cross-Encoder 점수
    metadata: dict

    def summary(self) -> str:
        return (
            f"[{self.metadata.get('title', '제목 없음')}]\n"
            f"기관: {self.metadata.get('issuing_org', '-')} | "
            f"도메인: {self.metadata.get('domain', '-')} | "
            f"예산: {self.metadata.get('budget_min', 0)/10000:.0f}억~"
            f"{self.metadata.get('budget_max', 0)/10000:.0f}억원 | "
            f"마감: {self.metadata.get('deadline', '-')}\n"
            f"섹션: {self.section}\n"
            f"{self.text[:200]}..."
        )


# ─────────────────────────────────────────
# 2. 메타데이터 필터 빌더
# ─────────────────────────────────────────

class MetadataFilterBuilder:
    """
    ChromaDB where 절 구문으로 복잡한 메타데이터 필터를 빌드합니다.

    ChromaDB 지원 연산자:
      $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
      $and, $or

    예시:
        filter = (MetadataFilterBuilder()
            .org_type("government")
            .budget_range(min_budget=30000, max_budget=200000)
            .domain("IT")
            .deadline_before("2024-06-30")
            .build())
    """

    def __init__(self):
        self._conditions: list[dict] = []

    def org_type(self, org_type: str) -> "MetadataFilterBuilder":
        """기관 유형: government / enterprise / public"""
        self._conditions.append({"org_type": {"$eq": org_type}})
        return self

    def issuing_org(self, org_name: str) -> "MetadataFilterBuilder":
        """특정 발주 기관"""
        self._conditions.append({"issuing_org": {"$eq": org_name}})
        return self

    def budget_range(
        self,
        min_budget: Optional[float] = None,
        max_budget: Optional[float] = None,
    ) -> "MetadataFilterBuilder":
        """
        예산 범위 필터 (단위: 만원)
        - 문서의 budget_max >= 요청 min_budget (너무 작은 사업 제외)
        - 문서의 budget_min <= 요청 max_budget (너무 큰 사업 제외)
        """
        if min_budget is not None:
            self._conditions.append({"budget_max": {"$gte": float(min_budget)}})
        if max_budget is not None:
            self._conditions.append({"budget_min": {"$lte": float(max_budget)}})
        return self

    def domain(self, domain: str) -> "MetadataFilterBuilder":
        """사업 도메인: IT / 건설 / 컨설팅 / 연구개발 등"""
        self._conditions.append({"domain": {"$eq": domain}})
        return self

    def domains(self, domains: list[str]) -> "MetadataFilterBuilder":
        """복수 도메인 중 하나"""
        self._conditions.append({"domain": {"$in": domains}})
        return self

    def region(self, region: str) -> "MetadataFilterBuilder":
        """지역 필터"""
        self._conditions.append({"region": {"$eq": region}})
        return self

    def deadline_before(self, date_str: str) -> "MetadataFilterBuilder":
        """마감일이 특정 날짜 이전인 RFP (YYYY-MM-DD 형식으로 lexicographic 비교)"""
        self._conditions.append({"deadline": {"$lte": date_str}})
        return self

    def deadline_after(self, date_str: str) -> "MetadataFilterBuilder":
        """마감일이 특정 날짜 이후인 RFP"""
        self._conditions.append({"deadline": {"$gte": date_str}})
        return self

    def submission_method(self, method: str) -> "MetadataFilterBuilder":
        """제출 방식: 온라인 / 방문 / 우편"""
        self._conditions.append({"submission_method": {"$eq": method}})
        return self

    def doc_ids(self, doc_ids: list[str]) -> "MetadataFilterBuilder":
        """특정 문서 ID 범위 내에서만 검색"""
        self._conditions.append({"doc_id": {"$in": doc_ids}})
        return self

    def build(self) -> Optional[dict]:
        """ChromaDB where 절 생성"""
        if not self._conditions:
            return None
        if len(self._conditions) == 1:
            return self._conditions[0]
        return {"$and": self._conditions}

    def build_or(self) -> Optional[dict]:
        """OR 조건으로 결합"""
        if not self._conditions:
            return None
        if len(self._conditions) == 1:
            return self._conditions[0]
        return {"$or": self._conditions}


# ─────────────────────────────────────────
# 3. BM25 스파스 검색기
# ─────────────────────────────────────────

class BM25Retriever:
    """
    ChromaDB에서 가져온 후보 청크에 BM25 적용.
    한국어 형태소 분리를 위해 간단한 음절 기반 토크나이저 사용
    (실서비스에서는 KoNLPy/Mecab 권장).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._index: Optional[BM25Okapi] = None
        self._doc_ids: list[str] = []

    def tokenize(self, text: str) -> list[str]:
        """
        간단한 공백+음절 토크나이저.
        실서비스 권장: konlpy.tag.Mecab().morphs(text)
        """
        # 특수문자 제거 후 공백 분리
        text = re.sub(r"[^\w가-힣a-zA-Z0-9]", " ", text)
        tokens = text.split()
        # 2글자 이상 토큰만 유지
        return [t for t in tokens if len(t) >= 2]

    def build_index(self, texts: list[str], chunk_ids: list[str]) -> None:
        """주어진 텍스트로 BM25 인덱스 구축"""
        self._doc_ids = chunk_ids
        tokenized = [self.tokenize(t) for t in texts]
        self._index = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def get_scores(self, query: str) -> dict[str, float]:
        """쿼리에 대한 각 청크의 BM25 점수 반환"""
        if self._index is None:
            return {}
        query_tokens = self.tokenize(query)
        scores = self._index.get_scores(query_tokens)
        return {cid: float(s) for cid, s in zip(self._doc_ids, scores)}


# ─────────────────────────────────────────
# 4. 쿼리 확장
# ─────────────────────────────────────────

class RFPQueryExpander:
    """
    RFP 도메인 특화 쿼리 확장.
    동의어/유사어를 추가하여 검색 recall을 향상시킵니다.
    실서비스에서는 LLM API 호출로 대체 권장.
    """

    SYNONYM_MAP = {
        "예산": ["사업비", "총사업비", "계약금액", "용역비", "비용"],
        "제출": ["납품", "접수", "제안", "입찰"],
        "마감": ["기한", "deadline", "제출기한", "접수마감"],
        "발주": ["공고", "입찰공고", "사업공고"],
        "IT": ["정보기술", "소프트웨어", "SW", "ICT", "디지털"],
        "컨설팅": ["용역", "자문", "advisory", "컨설턴시"],
        "클라우드": ["cloud", "SaaS", "PaaS", "IaaS"],
        "보안": ["security", "정보보호", "사이버보안"],
        "AI": ["인공지능", "머신러닝", "딥러닝", "ML"],
    }

    def expand(self, query: str, max_expansions: int = 3) -> str:
        """쿼리에 동의어를 추가하여 확장된 쿼리 반환"""
        expanded_terms = [query]
        words = query.split()

        for word in words:
            for key, synonyms in self.SYNONYM_MAP.items():
                if key in word or word in synonyms:
                    expanded_terms.extend(synonyms[:max_expansions])
                    break

        # 중복 제거 후 결합
        seen = set()
        unique_terms = []
        for t in expanded_terms:
            if t not in seen:
                seen.add(t)
                unique_terms.append(t)

        return " ".join(unique_terms)


# ─────────────────────────────────────────
# 5. Cross-Encoder 리랭커
# ─────────────────────────────────────────

class CrossEncoderReranker:
    """
    Bi-Encoder 검색 결과를 Cross-Encoder로 재순위화합니다.
    정밀도를 크게 향상시키지만 속도가 느리므로 Top-K에만 적용합니다.
    
    권장 모델:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (영어, 빠름)
    - Dongjin-kr/ko-reranker (한국어 특화)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[Reranker] Cross-Encoder 로딩: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """결과 리스트를 Cross-Encoder 점수로 재정렬"""
        if not results:
            return results

        # (query, passage) 쌍 생성
        pairs = [(query, r.text) for r in results]
        scores = self.model.predict(pairs)

        # 점수 업데이트 후 정렬
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
            result.score = float(score)  # 최종 점수를 rerank 점수로 교체

        reranked = sorted(results, key=lambda x: x.rerank_score or 0, reverse=True)
        return reranked[:top_k]


# ─────────────────────────────────────────
# 6. MMR (Maximal Marginal Relevance)
# ─────────────────────────────────────────

class MMRSelector:
    """
    검색 결과의 다양성을 보장하는 MMR 알고리즘.
    같은 문서에서 중복 청크가 반환되는 것을 방지합니다.
    
    MMR Score = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))
    """

    def select(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        candidates: list[RetrievalResult],
        top_k: int = 5,
        lambda_param: float = 0.7,  # 0=다양성 최대, 1=관련성 최대
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        q = np.array(query_embedding)
        C = np.array(candidate_embeddings)

        # 쿼리-후보 유사도
        query_sims = C @ q  # (N,)

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break

            best_idx = None
            best_mmr = -np.inf

            for idx in remaining:
                rel_score = query_sims[idx]

                if selected_indices:
                    selected_vecs = C[selected_indices]
                    redundancy = np.max(selected_vecs @ C[idx])
                else:
                    redundancy = 0.0

                mmr = lambda_param * rel_score - (1 - lambda_param) * redundancy

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining.remove(best_idx)
            candidates[best_idx].score = float(best_mmr)

        return [candidates[i] for i in selected_indices]


# ─────────────────────────────────────────
# 7. 통합 Retriever
# ─────────────────────────────────────────

class RFPRetriever:
    """
    고도화된 RFP 검색기.
    Dense + BM25 Hybrid → Metadata Filter → Rerank → MMR 파이프라인
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embed_model_name: str = "jhgan/ko-sroberta-multitask",
        reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        # 컴포넌트 초기화
        self.embedder = EmbeddingModel(embed_model_name)
        self.query_expander = RFPQueryExpander()
        self.bm25 = BM25Retriever()
        self.mmr_selector = MMRSelector()

        self.reranker = (
            CrossEncoderReranker(reranker_model) if reranker_model else None
        )

        # ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection("rfp_chunks")

        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def search(
        self,
        query: str,
        metadata_filter: Optional[dict] = None,
        top_k: int = 5,
        candidate_k: int = 30,
        use_rerank: bool = True,
        use_mmr: bool = True,
        expand_query: bool = True,
        mmr_lambda: float = 0.7,
    ) -> list[RetrievalResult]:
        """
        통합 검색 메서드.

        Args:
            query: 검색 쿼리
            metadata_filter: MetadataFilterBuilder.build() 결과
            top_k: 최종 반환할 결과 수
            candidate_k: 초기 후보 수 (rerank/MMR 전)
            use_rerank: Cross-Encoder 재순위화 사용 여부
            use_mmr: MMR 다양성 선택 사용 여부
            expand_query: 쿼리 확장 사용 여부
            mmr_lambda: MMR λ 파라미터 (0=다양성, 1=관련성)
        """

        # ── Step 1: 쿼리 확장 ──
        search_query = self.query_expander.expand(query) if expand_query else query
        print(f"[Search] 원본 쿼리: {query}")
        if expand_query:
            print(f"[Search] 확장 쿼리: {search_query}")

        # ── Step 2: Dense 검색 ──
        query_emb = self.embedder.encode_query(search_query)
        chroma_results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(candidate_k, self.collection.count() or 1),
            where=metadata_filter,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        if not chroma_results["ids"][0]:
            print("[Search] 검색 결과 없음")
            return []

        # ── Step 3: 결과 파싱 + BM25 스코어링 ──
        ids = chroma_results["ids"][0]
        docs = chroma_results["documents"][0]
        metas = chroma_results["metadatas"][0]
        distances = chroma_results["distances"][0]
        embeddings = chroma_results["embeddings"][0]

        # 코사인 거리 → 유사도 변환
        dense_scores = [1 - d for d in distances]

        # BM25 인덱스 구축 (후보 풀 내에서)
        self.bm25.build_index(docs, ids)
        bm25_scores = self.bm25.get_scores(search_query)

        # BM25 정규화 (0~1 범위)
        bm25_vals = list(bm25_scores.values())
        bm25_max = max(bm25_vals) if bm25_vals else 1.0
        bm25_max = bm25_max if bm25_max > 0 else 1.0

        # ── Step 4: Hybrid 점수 계산 ──
        results: list[RetrievalResult] = []
        for i, (chunk_id, doc, meta) in enumerate(zip(ids, docs, metas)):
            d_score = dense_scores[i]
            b_score = bm25_scores.get(chunk_id, 0.0) / bm25_max
            hybrid = self.dense_weight * d_score + self.bm25_weight * b_score

            results.append(RetrievalResult(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                text=doc,
                section=meta.get("section", ""),
                score=hybrid,
                dense_score=d_score,
                bm25_score=b_score,
                rerank_score=None,
                metadata=meta,
            ))

        # Hybrid 점수로 정렬
        results.sort(key=lambda x: x.score, reverse=True)
        print(f"[Search] Dense+BM25 후보: {len(results)}개")

        # ── Step 5: Cross-Encoder Reranking ──
        if use_rerank and self.reranker:
            results = self.reranker.rerank(query, results, top_k=candidate_k)
            print(f"[Search] Reranking 완료: {len(results)}개")

        # ── Step 6: MMR 다양성 선택 ──
        if use_mmr and embeddings:
            results = self.mmr_selector.select(
                query_embedding=query_emb,
                candidate_embeddings=embeddings[:len(results)],
                candidates=results,
                top_k=top_k,
                lambda_param=mmr_lambda,
            )
            print(f"[Search] MMR 선택 완료: {len(results)}개")
        else:
            results = results[:top_k]

        return results

    def search_by_budget(
        self,
        query: str,
        min_budget_만원: float,
        max_budget_만원: float,
        **kwargs,
    ) -> list[RetrievalResult]:
        """예산 범위 기반 빠른 검색 헬퍼"""
        f = (MetadataFilterBuilder()
             .budget_range(min_budget_만원, max_budget_만원)
             .build())
        return self.search(query, metadata_filter=f, **kwargs)

    def search_government_rfps(
        self,
        query: str,
        domain: Optional[str] = None,
        **kwargs,
    ) -> list[RetrievalResult]:
        """정부/공공기관 RFP 전용 검색"""
        builder = MetadataFilterBuilder().org_type("government")
        if domain:
            builder = builder.domain(domain)
        return self.search(query, metadata_filter=builder.build(), **kwargs)


# ─────────────────────────────────────────
# 8. 사용 예시
# ─────────────────────────────────────────

if __name__ == "__main__":
    retriever = RFPRetriever(
        persist_dir="./chroma_db",
        embed_model_name="jhgan/ko-sroberta-multitask",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        dense_weight=0.6,
        bm25_weight=0.4,
    )

    # ── 예시 1: 기본 시맨틱 검색 ──
    print("=" * 60)
    print("예시 1: 클라우드 전환 관련 IT RFP 검색")
    results = retriever.search(
        query="클라우드 전환 컨설팅",
        top_k=5,
        use_rerank=True,
        use_mmr=True,
    )
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] 점수={r.score:.3f} | Dense={r.dense_score:.3f} | BM25={r.bm25_score:.3f}")
        print(r.summary())

    # ── 예시 2: 메타데이터 필터 + 검색 ──
    print("\n" + "=" * 60)
    print("예시 2: 예산 5억~15억 정부 IT RFP")
    meta_filter = (
        MetadataFilterBuilder()
        .org_type("government")
        .domain("IT")
        .budget_range(min_budget=50000, max_budget=150000)
        .deadline_after("2024-01-01")
        .build()
    )
    results2 = retriever.search(
        query="보안 강화 시스템 구축",
        metadata_filter=meta_filter,
        top_k=5,
    )
    for i, r in enumerate(results2, 1):
        print(f"\n[{i}] {r.metadata.get('title', '-')}")
        print(f"    예산: {r.metadata.get('budget_min', 0)/10000:.0f}억~{r.metadata.get('budget_max', 0)/10000:.0f}억")

    # ── 예시 3: 헬퍼 메서드 ──
    print("\n" + "=" * 60)
    print("예시 3: 정부 RFP 전용 검색")
    results3 = retriever.search_government_rfps(
        query="데이터 거버넌스 체계 구축",
        domain="IT",
        top_k=3,
    )
    for r in results3:
        print(f"\n{r.summary()}")




    # ──────────────────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────────────
    # consulting
    # ──────────────────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────────────


"""
RFP 요약 및 컨설팅 추천 모듈
- 검색된 청크를 기반으로 핵심 정보 추출 및 요약
- 고객사 프로필에 맞는 RFP 추천 점수 산정
- 제안서 작성 시 주의사항 자동 도출
"""

import os
import json
import re
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

from rfp_retrieval import RFPRetriever, RetrievalResult, MetadataFilterBuilder


# ─────────────────────────────────────────
# 1. 고객사 프로필
# ─────────────────────────────────────────

@dataclass
class ClientProfile:
    """컨설팅 고객사 정보"""
    company_name: str
    domains: list[str]           # 전문 도메인
    regions: list[str]           # 활동 지역
    min_budget_만원: float       # 수용 가능 최소 예산
    max_budget_만원: float       # 수용 가능 최대 예산
    preferred_org_types: list[str]  # government / enterprise / public
    capabilities: list[str]      # 보유 역량 키워드
    past_clients: list[str]      # 과거 수행 기관 (레퍼런스)


# ─────────────────────────────────────────
# 2. 구조화된 RFP 요약
# ─────────────────────────────────────────

@dataclass
class RFPSummary:
    doc_id: str
    title: str
    issuing_org: str
    key_requirements: list[str]   # 핵심 요구 조건
    budget_summary: str           # 예산 요약
    deadline: str
    submission_method: str
    evaluation_criteria: list[str]  # 평가 기준
    risks: list[str]              # 잠재 리스크
    opportunities: list[str]     # 비즈니스 기회
    match_score: float            # 고객사 적합도 (0~100)
    recommendation: str           # 추천 여부 및 이유


# ─────────────────────────────────────────
# 3. LLM 기반 요약 엔진
# ─────────────────────────────────────────

class RFPSummarizer:
    """
    검색된 청크들을 LLM으로 요약합니다.
    OpenAI GPT-4o 또는 Claude API 사용 권장.
    """

    SYSTEM_PROMPT = """당신은 RFP(제안 요청서) 분석 전문가입니다.
제공된 RFP 텍스트를 분석하여 핵심 정보를 정확하고 간결하게 추출하세요.
반드시 JSON 형식으로만 응답하며, 추측이나 창작 없이 문서에 명시된 내용만 사용하세요."""

    EXTRACTION_PROMPT = """다음 RFP 텍스트를 분석하여 아래 JSON 형식으로 핵심 정보를 추출하세요:

RFP 텍스트:
{context}

추출 항목:
{{
  "key_requirements": ["핵심 요구사항 1", "핵심 요구사항 2", ...],
  "budget_summary": "예산 관련 핵심 내용을 1~2문장으로",
  "evaluation_criteria": ["평가 기준 1", "평가 기준 2", ...],
  "risks": ["수주 시 리스크 1", "리스크 2", ...],
  "opportunities": ["비즈니스 기회 1", "기회 2", ...],
  "submission_requirements": ["제출 필수 서류 1", "서류 2", ...]
}}

JSON만 출력하세요. 마크다운 코드 블록 없이."""

    RECOMMENDATION_PROMPT = """다음 RFP 정보와 고객사 프로필을 비교하여 추천 분석을 작성하세요.

RFP 요약:
{rfp_summary}

고객사 프로필:
- 회사명: {company_name}
- 전문 도메인: {domains}
- 보유 역량: {capabilities}
- 예산 범위: {budget_range}만원
- 과거 수행 기관: {past_clients}

다음 JSON 형식으로 응답하세요:
{{
  "match_score": 0~100 사이 정수,
  "match_reasons": ["적합 이유 1", "이유 2", ...],
  "concerns": ["우려 사항 1", "사항 2", ...],
  "recommendation": "적극 추천 / 조건부 추천 / 비추천",
  "action_items": ["제안서 작성 시 주의사항 1", "사항 2", ...]
}}

JSON만 출력하세요."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        """LLM API 호출"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()

    def _safe_json_parse(self, text: str) -> dict:
        """LLM 응답에서 JSON 안전하게 파싱"""
        # ```json ... ``` 블록 제거
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # JSON 추출 시도
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {}

    def extract_rfp_info(self, chunks: list[RetrievalResult]) -> dict:
        """검색된 청크에서 RFP 핵심 정보 추출"""
        # 상위 청크 컨텍스트 결합 (토큰 제한 고려)
        context = "\n\n---\n\n".join(
            f"[{c.section}]\n{c.text}" for c in chunks[:8]
        )
        prompt = self.EXTRACTION_PROMPT.format(context=context)
        response = self._call_llm(prompt)
        return self._safe_json_parse(response)

    def generate_recommendation(
        self,
        rfp_info: dict,
        metadata: dict,
        client: ClientProfile,
    ) -> dict:
        """고객사 프로필 기반 추천 분석 생성"""
        rfp_summary_str = json.dumps(
            {**rfp_info, **{k: metadata.get(k, "") for k in
             ["title", "issuing_org", "budget_min", "budget_max", "deadline"]}},
            ensure_ascii=False, indent=2
        )
        prompt = self.RECOMMENDATION_PROMPT.format(
            rfp_summary=rfp_summary_str,
            company_name=client.company_name,
            domains=", ".join(client.domains),
            capabilities=", ".join(client.capabilities),
            budget_range=f"{client.min_budget_만원:,.0f}~{client.max_budget_만원:,.0f}",
            past_clients=", ".join(client.past_clients),
        )
        response = self._call_llm(prompt, temperature=0.2)
        return self._safe_json_parse(response)

    def summarize_rfp(
        self,
        chunks: list[RetrievalResult],
        client: ClientProfile,
    ) -> RFPSummary:
        """전체 요약 파이프라인: 정보 추출 → 추천 분석 → RFPSummary 반환"""
        if not chunks:
            raise ValueError("검색 결과가 없습니다.")

        meta = chunks[0].metadata

        # 1) RFP 핵심 정보 추출
        print("[Summarizer] RFP 핵심 정보 추출 중...")
        rfp_info = self.extract_rfp_info(chunks)

        # 2) 고객사 적합도 분석
        print("[Summarizer] 고객사 적합도 분석 중...")
        recommendation = self.generate_recommendation(rfp_info, meta, client)

        return RFPSummary(
            doc_id=meta.get("doc_id", ""),
            title=meta.get("title", ""),
            issuing_org=meta.get("issuing_org", ""),
            key_requirements=rfp_info.get("key_requirements", []),
            budget_summary=rfp_info.get("budget_summary", ""),
            deadline=meta.get("deadline", ""),
            submission_method=meta.get("submission_method", ""),
            evaluation_criteria=rfp_info.get("evaluation_criteria", []),
            risks=rfp_info.get("risks", []),
            opportunities=rfp_info.get("opportunities", []),
            match_score=recommendation.get("match_score", 0),
            recommendation=recommendation.get("recommendation", ""),
        )


# ─────────────────────────────────────────
# 4. 컨설팅 추천 엔진 (통합)
# ─────────────────────────────────────────

class RFPConsultingEngine:
    """
    고객사에 맞는 RFP를 검색하고, 요약 및 추천 분석까지 제공하는
    엔드-투-엔드 컨설팅 엔진.
    """

    def __init__(
        self,
        retriever: RFPRetriever,
        summarizer: RFPSummarizer,
    ):
        self.retriever = retriever
        self.summarizer = summarizer

    def recommend_for_client(
        self,
        client: ClientProfile,
        query: Optional[str] = None,
        top_n: int = 3,
    ) -> list[RFPSummary]:
        """
        고객사 프로필 기반으로 최적 RFP를 검색하고 요약 추천을 생성합니다.
        """
        # 쿼리 자동 생성 (미입력 시)
        if not query:
            query = " ".join(client.domains + client.capabilities[:3])

        # 메타데이터 필터 빌드
        meta_filter = (
            MetadataFilterBuilder()
            .domains(client.domains)
            .budget_range(client.min_budget_만원, client.max_budget_만원)
            .build()
        )

        # 검색
        print(f"\n[Engine] '{client.company_name}' 맞춤 RFP 검색 중...")
        results = self.retriever.search(
            query=query,
            metadata_filter=meta_filter,
            top_k=top_n * 3,  # 요약 후 필터링 여유분
            use_rerank=True,
            use_mmr=True,
            mmr_lambda=0.65,
        )

        # 문서별로 청크 그룹화 (같은 doc_id 청크 묶기)
        doc_chunks: dict[str, list[RetrievalResult]] = {}
        for r in results:
            doc_chunks.setdefault(r.doc_id, []).append(r)

        # 각 문서 요약 및 추천 분석
        summaries: list[RFPSummary] = []
        for doc_id, chunks in list(doc_chunks.items())[:top_n]:
            print(f"\n[Engine] 문서 요약 중: {doc_id}")
            try:
                summary = self.summarizer.summarize_rfp(chunks, client)
                summaries.append(summary)
            except Exception as e:
                print(f"  ⚠ 요약 실패: {e}")

        # 매칭 점수 내림차순 정렬
        summaries.sort(key=lambda s: s.match_score, reverse=True)
        return summaries

    def print_recommendation_report(
        self,
        summaries: list[RFPSummary],
        client: ClientProfile,
    ) -> None:
        """추천 결과를 콘솔에 포맷팅하여 출력"""
        print("\n" + "=" * 70)
        print(f"  RFP 컨설팅 추천 리포트 - {client.company_name}")
        print("=" * 70)

        for i, s in enumerate(summaries, 1):
            bar = "█" * int(s.match_score / 10) + "░" * (10 - int(s.match_score / 10))
            print(f"\n{'─'*70}")
            print(f"#{i} [{s.recommendation}] 적합도: {s.match_score}/100 |{bar}|")
            print(f"   제목: {s.title}")
            print(f"   발주: {s.issuing_org}")
            print(f"   예산: {s.budget_summary}")
            print(f"   마감: {s.deadline} | 제출: {s.submission_method}")

            print("\n   📋 핵심 요구사항:")
            for req in s.key_requirements[:5]:
                print(f"      • {req}")

            print("\n   ⚖️  평가 기준:")
            for crit in s.evaluation_criteria[:3]:
                print(f"      • {crit}")

            print("\n   ⚠️  리스크:")
            for risk in s.risks[:3]:
                print(f"      • {risk}")

            print("\n   💡 비즈니스 기회:")
            for opp in s.opportunities[:3]:
                print(f"      • {opp}")

        print("\n" + "=" * 70)


# ─────────────────────────────────────────
# 5. 사용 예시
# ─────────────────────────────────────────

if __name__ == "__main__":
    # 고객사 프로필 정의
    client = ClientProfile(
        company_name="테크솔루션즈 주식회사",
        domains=["IT", "컨설팅"],
        regions=["서울", "경기", "전국"],
        min_budget_만원=30_000,       # 3억원
        max_budget_만원=200_000,      # 20억원
        preferred_org_types=["government", "public"],
        capabilities=["클라우드 전환", "데이터 거버넌스", "보안 컨설팅", "AI/ML"],
        past_clients=["한국정보화진흥원", "금융감독원", "국민건강보험공단"],
    )

    # Retriever 초기화
    retriever = RFPRetriever(
        persist_dir="./chroma_db",
        embed_model_name="jhgan/ko-sroberta-multitask",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    # Summarizer 초기화 (OPENAI_API_KEY 환경변수 필요)
    summarizer = RFPSummarizer(model="gpt-4o")

    # 컨설팅 엔진
    engine = RFPConsultingEngine(retriever, summarizer)

    # 추천 실행
    summaries = engine.recommend_for_client(
        client=client,
        query="공공기관 디지털 전환 클라우드 보안",
        top_n=3,
    )

    # 리포트 출력
    engine.print_recommendation_report(summaries, client)


