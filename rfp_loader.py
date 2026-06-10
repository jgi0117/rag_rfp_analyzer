"""
RFP Document Loading Pipeline
==============================
실제 데이터 구조 기반:
  - 정제된 CSV  : 공고 번호, 사업명, 사업 금액, 발주 기관, 날짜, 사업 요약, 텍스트(HWP 원문) 등 12개 컬럼
  - 원본 파일    : HWP 96건 / PDF 4건 (향후 확장 가능)
  - 출력        : RAG 인덱싱용 DocumentChunk 리스트 + JSONL

HWP 처리 전략 (우선순위):
  1. LibreOffice  HWP -> PDF 변환 후 pdfplumber 로 추출  [메인]
     - 한글 설치 불필요, 서버 환경에서 가장 안정적
     - sudo apt install libreoffice
  2. pyhwp (hwp5txt CLI)                                  [보조]
     - pip install pyhwp
  3. 바이너리 직접 파싱                                    [최후 수단]
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber
from pypdf import PdfReader

# ── 로깅 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rfp_pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 클래스
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RFPMeta:
    """
    CSV 한 행 = 하나의 RFP 공고 메타데이터
    벡터 DB 필터링/정렬에 활용됩니다.
    """
    공고번호: str
    공고차수: Optional[float]
    사업명: str
    사업금액: Optional[float]        # 원 단위
    발주기관: str
    공개일자: Optional[str]
    입찰시작일: Optional[str]
    입찰마감일: Optional[str]
    파일형식: str                    # "hwp" | "pdf"
    파일명: str


@dataclass
class DocumentChunk:
    """RAG 인덱싱 단위 — 벡터 DB 1 레코드"""
    chunk_id: str           # "{공고번호}_chunk_{n}"
    공고번호: str
    사업명: str
    발주기관: str
    사업금액: Optional[float]
    입찰마감일: Optional[str]
    파일형식: str
    text_source: str        # "csv_summary" | "csv_fulltext" | "file_extract"
    text: str
    metadata: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 텍스트 청커 (슬라이딩 윈도우)
# ══════════════════════════════════════════════════════════════════════════════

class TextChunker:
    """
    RFP 조항 경계를 보존하기 위해 overlap 포함 슬라이딩 윈도우 사용.
    chunk_size=500 / overlap=100 이 RFP 문서에 적합합니다.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(
        self,
        text: str,
        chunk_id_prefix: str,
        text_source: str,
        meta: RFPMeta,
    ) -> list[DocumentChunk]:
        words = text.split()
        chunks: list[DocumentChunk] = []
        start, idx = 0, 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{chunk_id_prefix}_chunk_{idx}",
                    공고번호=meta.공고번호,
                    사업명=meta.사업명,
                    발주기관=meta.발주기관,
                    사업금액=meta.사업금액,
                    입찰마감일=meta.입찰마감일,
                    파일형식=meta.파일형식,
                    text_source=text_source,
                    text=" ".join(words[start:end]),
                )
            )
            if end == len(words):
                break
            start = end - self.overlap
            idx += 1

        return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 3. CSV 로더  (현재 메인 소스)
# ══════════════════════════════════════════════════════════════════════════════

class CSVLoader:
    """
    정제된 CSV를 읽어 청크를 생성합니다.

    텍스트 우선순위:
      1. '텍스트' 컬럼 (HWP 원문 — 가장 상세)
      2. '사업 요약' 컬럼 (LLM 요약본 — 텍스트가 없을 때 보완)
    두 컬럼 모두 별도 청크로 만들어 RAG 검색 품질을 높입니다.
    """

    def __init__(self, chunker: TextChunker, encoding: str = "utf-8"):
        self.chunker = chunker
        self.encoding = encoding

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """불필요한 공백/이스케이프 시퀀스 정리"""
        text = text.replace("\\n", "\n").replace("\\r", "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _row_to_meta(self, row: pd.Series) -> RFPMeta:
        def safe(val) -> Optional[str]:
            s = str(val)
            return None if s in ("nan", "None", "") else s

        return RFPMeta(
            공고번호=str(row["공고 번호"]),
            공고차수=row.get("공고 차수"),
            사업명=str(row.get("사업명", "")),
            사업금액=row.get("사업 금액"),
            발주기관=str(row.get("발주 기관", "")),
            공개일자=safe(row.get("공개 일자")),
            입찰시작일=safe(row.get("입찰 참여 시작일")),
            입찰마감일=safe(row.get("입찰 참여 마감일")),
            파일형식=str(row.get("파일형식", "hwp")).lower(),
            파일명=str(row.get("파일명", "")),
        )

    # ── 공개 메서드 ───────────────────────────────────────────────────────────

    def load(self, csv_path: str) -> list[DocumentChunk]:
        logger.info(f"[CSV] 로딩 시작: {csv_path}")
        df = pd.read_csv(csv_path, encoding=self.encoding)
        logger.info(f"[CSV] {len(df)}행 로드 완료")

        all_chunks: list[DocumentChunk] = []

        for _, row in df.iterrows():
            meta = self._row_to_meta(row)

            # ① 원문 텍스트 청킹 (가장 중요)
            raw_text = str(row.get("텍스트", ""))
            if raw_text and raw_text != "nan":
                cleaned = self._clean_text(raw_text)
                all_chunks.extend(
                    self.chunker.split(cleaned, f"{meta.공고번호}_full", "csv_fulltext", meta)
                )

            # ② 사업 요약 청킹 (짧고 구조적 — 검색 recall 보완)
            summary = str(row.get("사업 요약", ""))
            if summary and summary != "nan":
                cleaned_summary = self._clean_text(summary)
                all_chunks.extend(
                    self.chunker.split(cleaned_summary, f"{meta.공고번호}_summ", "csv_summary", meta)
                )

        logger.info(f"[CSV] 총 {len(all_chunks)}개 청크 생성")
        return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# 4. HWP 파일 로더
# ══════════════════════════════════════════════════════════════════════════════

class HWPLoader:
    """
    HWP 원본 파일 → 텍스트 추출.

    추출 전략 (우선순위):
      1. LibreOffice  HWP -> PDF 변환 후 pdfplumber  [메인, 가장 안정적]
      2. pyhwp (hwp5txt CLI)                          [보조]
      3. 바이너리 직접 파싱                            [최후 수단]

    LibreOffice 설치:
      Ubuntu/Debian  : sudo apt install libreoffice
      macOS          : brew install --cask libreoffice
      Docker         : RUN apt-get install -y libreoffice
    """

    # LibreOffice 실행 파일 후보 (OS마다 경로가 다름)
    _LO_CANDIDATES = ["libreoffice", "soffice", "/usr/bin/libreoffice", "/usr/bin/soffice"]

    def __init__(self, chunker: TextChunker, lo_timeout: int = 120):
        self.chunker    = chunker
        self.lo_timeout = lo_timeout
        self._lo_bin    = self._find_libreoffice()

    # ── LibreOffice 경로 탐색 ─────────────────────────────────────────────────

    def _find_libreoffice(self) -> Optional[str]:
        for candidate in self._LO_CANDIDATES:
            if shutil.which(candidate):
                logger.info(f"[HWP] LibreOffice 발견: {candidate}")
                return candidate
        logger.warning("[HWP] LibreOffice 미설치 → 보조 전략 사용 (sudo apt install libreoffice)")
        return None

    # ── 전략 1: LibreOffice HWP → PDF → pdfplumber ───────────────────────────

    def _extract_via_libreoffice(self, hwp_path: str) -> Optional[str]:
        """
        HWP를 임시 디렉토리에서 PDF로 변환한 뒤 pdfplumber로 텍스트를 읽습니다.
        임시 파일은 성공/실패 무관하게 반드시 정리됩니다.
        """
        if not self._lo_bin:
            return None

        tmp_dir = tempfile.mkdtemp(prefix="rfp_lo_")
        try:
            # LibreOffice는 --outdir 경로에 동일한 파일명의 .pdf를 생성합니다
            result = subprocess.run(
                [
                    self._lo_bin,
                    "--headless",
                    "--norestore",          # 복구 대화상자 방지
                    "--convert-to", "pdf",
                    "--outdir", tmp_dir,
                    hwp_path,
                ],
                capture_output=True,
                timeout=self.lo_timeout,
            )

            if result.returncode != 0:
                logger.warning(
                    f"[HWP] LibreOffice 변환 실패 (code={result.returncode}): "
                    f"{result.stderr.decode(errors='ignore')[:200]}"
                )
                return None

            pdf_files = list(Path(tmp_dir).glob("*.pdf"))
            if not pdf_files:
                logger.warning("[HWP] LibreOffice 변환 후 PDF 파일 없음")
                return None

            return self._read_pdf(str(pdf_files[0]))

        except subprocess.TimeoutExpired:
            logger.warning(f"[HWP] LibreOffice 타임아웃 ({self.lo_timeout}s): {hwp_path}")
            return None
        except Exception as e:
            logger.warning(f"[HWP] LibreOffice 오류: {e}")
            return None
        finally:
            # 임시 디렉토리 반드시 정리
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _read_pdf(self, pdf_path: str) -> Optional[str]:
        """변환된 PDF에서 텍스트 추출 (pdfplumber 우선 / pypdf 폴백)"""
        try:
            pages = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    # 표 안의 텍스트도 함께 추출
                    for table in page.extract_tables():
                        for row in table:
                            text += "\n" + " | ".join(c or "" for c in row)
                    pages.append(text.strip())
            result = "\n\n".join(p for p in pages if p)
            return result if result.strip() else None
        except Exception:
            # pdfplumber 실패 시 pypdf 폴백
            try:
                reader = PdfReader(pdf_path)
                result = "\n\n".join(
                    p.extract_text() or "" for p in reader.pages
                )
                return result if result.strip() else None
            except Exception as e:
                logger.warning(f"[HWP] PDF 읽기 실패: {e}")
                return None

    # ── 전략 2: pyhwp (hwp5txt CLI) ──────────────────────────────────────────

    def _extract_via_pyhwp(self, hwp_path: str) -> Optional[str]:
        """pip install pyhwp 설치 시 사용 가능"""
        try:
            result = subprocess.run(
                ["hwp5txt", hwp_path],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
            logger.warning(f"[HWP] hwp5txt 실패: {result.stderr[:200]}")
            return None
        except FileNotFoundError:
            logger.warning("[HWP] hwp5txt 없음 (pip install pyhwp)")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("[HWP] hwp5txt 타임아웃")
            return None

    # ── 전략 3: 바이너리 직접 파싱 (최후 수단) ───────────────────────────────

    def _extract_binary_fallback(self, hwp_path: str) -> str:
        """
        HWP 바이너리에서 한글/ASCII 텍스트 패턴만 추출합니다.
        표·이미지 내 텍스트는 누락될 수 있으며, 품질이 낮습니다.
        """
        logger.warning(f"[HWP] 바이너리 파싱 사용 (품질 낮음): {hwp_path}")
        try:
            raw  = Path(hwp_path).read_bytes()
            text = raw.decode("utf-16-le", errors="ignore")
            # 한글 + 기본 ASCII만 남기고 제어문자 제거
            text = re.sub(r"[^\uAC00-\uD7A3\u0020-\u007E\n]", " ", text)
            text = re.sub(r" {3,}", "  ", text)
            return text.strip()
        except Exception as e:
            logger.error(f"[HWP] 바이너리 파싱 실패: {e}")
            return ""

    # ── 공개 메서드 ───────────────────────────────────────────────────────────

    def load(self, hwp_path: str, meta: RFPMeta) -> list[DocumentChunk]:
        """
        HWP 파일을 읽어 DocumentChunk 리스트를 반환합니다.
        전략 1(LibreOffice) → 2(pyhwp) → 3(바이너리) 순으로 시도합니다.
        """
        logger.info(f"[HWP] 로딩 시작: {hwp_path}")

        text = (
            self._extract_via_libreoffice(hwp_path)
            or self._extract_via_pyhwp(hwp_path)
            or self._extract_binary_fallback(hwp_path)
        )

        if not text or not text.strip():
            logger.error(f"[HWP] 모든 전략 실패, 청크 없음: {hwp_path}")
            return []

        logger.info(f"[HWP] 추출 완료: {len(text):,}자")
        return self.chunker.split(
            text,
            chunk_id_prefix=f"{meta.공고번호}_hwp",
            text_source="file_extract",
            meta=meta,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. PDF 파일 로더  (현재 4건, 향후 확장 대비)
# ══════════════════════════════════════════════════════════════════════════════

class PDFLoader:
    """
    PDF 원본 파일 텍스트 추출.

    추출 전략:
      1. pdfplumber (레이아웃 + 테이블 동시 추출)
      2. pypdf      (폴백)
      3. pytesseract OCR (스캔 PDF — 텍스트 30자 미만 페이지에 자동 적용)
    """

    def __init__(self, chunker: TextChunker):
        self.chunker = chunker

    def _pdfplumber(self, path: str) -> tuple[list[str], int]:
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for table in page.extract_tables():
                    for row in table:
                        text += "\n" + " | ".join(c or "" for c in row)
                pages.append(text.strip())
        return pages, len(pages)

    def _pypdf(self, path: str) -> tuple[list[str], int]:
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return pages, len(pages)

    def _ocr(self, path: str, page_num: int) -> str:
        try:
            import pytesseract
            from pdf2image import convert_from_path
            imgs = convert_from_path(path, first_page=page_num, last_page=page_num)
            return pytesseract.image_to_string(imgs[0], lang="kor+eng") if imgs else ""
        except ImportError:
            logger.warning("pytesseract/pdf2image 미설치 -> OCR 건너뜀")
            return ""

    def load(self, path: str, meta: RFPMeta) -> list[DocumentChunk]:
        logger.info(f"[PDF] 파싱: {path}")
        try:
            pages, _ = self._pdfplumber(path)
        except Exception:
            pages, _ = self._pypdf(path)

        all_chunks: list[DocumentChunk] = []
        for i, text in enumerate(pages, start=1):
            if len(text.strip()) < 30:          # 스캔 페이지 판단
                text = self._ocr(path, i) or text
            if not text.strip():
                continue
            all_chunks.extend(
                self.chunker.split(text, f"{meta.공고번호}_pdf_p{i}", "file_extract", meta)
            )
        return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# 6. 통합 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

class RFPPipeline:
    """
    CSV + 원본 파일(HWP/PDF)을 함께 읽어 RAG용 청크를 생성합니다.

    사용 시나리오
    ─────────────
    A. CSV만 있을 때         -> load_from_csv()              (현재 권장)
    B. CSV + 원본 파일 있을때 -> load_from_csv_and_files()
    C. 원본 파일만           -> load_file()
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunker  = TextChunker(chunk_size, overlap)
        self.csv_ldr  = CSVLoader(self.chunker)
        self.hwp_ldr  = HWPLoader(self.chunker)
        self.pdf_ldr  = PDFLoader(self.chunker)

    # ── A. CSV만 ──────────────────────────────────────────────────────────────

    def load_from_csv(self, csv_path: str) -> list[DocumentChunk]:
        """정제된 CSV에서 바로 청크를 생성합니다 (현재 권장 방식)."""
        return self.csv_ldr.load(csv_path)

    # ── B. CSV + 원본 파일 ────────────────────────────────────────────────────

    def load_from_csv_and_files(
        self, csv_path: str, file_dir: str
    ) -> list[DocumentChunk]:
        """
        CSV 메타데이터 + file_dir 원본 파일을 함께 파싱합니다.
        CSV '텍스트' 컬럼이 잘린 경우 원본 파일로 보완합니다.
        """
        df = pd.read_csv(csv_path, encoding="utf-8")
        all_chunks: list[DocumentChunk] = []

        for _, row in df.iterrows():
            meta = self.csv_ldr._row_to_meta(row)
            file_path = Path(file_dir) / meta.파일명

            if file_path.exists():
                if meta.파일형식 == "hwp":
                    all_chunks.extend(self.hwp_ldr.load(str(file_path), meta))
                elif meta.파일형식 == "pdf":
                    all_chunks.extend(self.pdf_ldr.load(str(file_path), meta))
            else:
                # 파일 없으면 CSV 텍스트 사용
                logger.warning(f"파일 없음 -> CSV 텍스트 사용: {file_path}")
                raw = str(row.get("텍스트", ""))
                if raw and raw != "nan":
                    all_chunks.extend(
                        self.chunker.split(
                            self.csv_ldr._clean_text(raw),
                            f"{meta.공고번호}_full",
                            "csv_fulltext",
                            meta,
                        )
                    )

        logger.info(f"[Pipeline] 총 {len(all_chunks)}개 청크 생성")
        return all_chunks

    # ── C. 단일 파일 ──────────────────────────────────────────────────────────

    def load_file(self, file_path: str, meta: Optional[RFPMeta] = None) -> list[DocumentChunk]:
        """원본 파일 하나를 직접 파싱합니다."""
        p = Path(file_path)
        if meta is None:
            meta = RFPMeta(
                공고번호=p.stem, 공고차수=None, 사업명=p.stem,
                사업금액=None, 발주기관="",
                공개일자=None, 입찰시작일=None, 입찰마감일=None,
                파일형식=p.suffix.lower().lstrip("."),
                파일명=p.name,
            )
        ext = p.suffix.lower()
        if ext in {".hwp", ".hwpx"}:
            return self.hwp_ldr.load(str(p), meta)
        elif ext == ".pdf":
            return self.pdf_ldr.load(str(p), meta)
        else:
            logger.warning(f"지원하지 않는 형식: {ext}")
            return []

    # ── 출력 ──────────────────────────────────────────────────────────────────

    def export_jsonl(self, chunks: list[DocumentChunk], out_path: str) -> None:
        """벡터 DB 인덱싱용 JSONL 저장"""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        logger.info(f"[Export] JSONL 저장: {out_path} ({len(chunks)}건)")

    def summary(self, chunks: list[DocumentChunk]) -> None:
        """청크 통계 출력"""
        from collections import Counter
        sources = Counter(c.text_source for c in chunks)
        types   = Counter(c.파일형식 for c in chunks)
        print("\n=== 청크 요약 ===")
        print(f"총 청크 수     : {len(chunks):,}")
        print(f"텍스트 출처    : {dict(sources)}")
        print(f"파일 형식      : {dict(types)}")
        avg = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
        print(f"평균 청크 길이 : {avg:.0f}자")


# ══════════════════════════════════════════════════════════════════════════════
# 7. 실행 예시
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = RFPPipeline(chunk_size=500, overlap=100)

    # ── 시나리오 A: CSV만 있을 때 (현재 권장) ────────────────────────────────
    CSV_PATH = "data_list.csv"
    chunks = pipeline.load_from_csv(CSV_PATH)
    pipeline.summary(chunks)
    pipeline.export_jsonl(chunks, "output/rfp_chunks.jsonl")

    # ── 시나리오 B: CSV + 원본 HWP/PDF 파일 ──────────────────────────────────
    # chunks = pipeline.load_from_csv_and_files(
    #     csv_path="data_list.csv",
    #     file_dir="./hwp_files",     # HWP / PDF 파일이 있는 폴더
    # )

    # ── 시나리오 C: 단일 파일 직접 파싱 ──────────────────────────────────────
    # chunks = pipeline.load_file("sample_rfp.hwp")
    # chunks = pipeline.load_file("sample_rfp.pdf")
