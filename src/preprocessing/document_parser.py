"""
Parse PDF documents with Docling and split Markdown by headings.

Examples:
    python src/preprocessing/document_parser.py --limit 3
    python src/preprocessing/document_parser.py --ocr --limit 3
    python src/preprocessing/document_parser.py --ocr --force-full-page-ocr --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "pdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "parsed"
SUPPORTED_EXTENSIONS = {".pdf"}


@dataclass
class MarkdownSection:
    section_id: str
    title: str
    level: int
    order: int
    content: str
    source_path: str


@dataclass
class ParseResult:
    source_path: Path
    markdown_path: Path
    metadata_path: Path
    sections_path: Path
    status: str
    elapsed_seconds: float
    text_length: int = 0
    section_count: int = 0
    message: str = ""


class DoclingParser:
    engine_name = "docling"

    def __init__(
        self,
        do_ocr: bool,
        force_full_page_ocr: bool,
        ocr_engine: str,
        ocr_lang: list[str] | None,
        do_table_structure: bool,
        normalize_pdf: bool,
    ) -> None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise RuntimeError(
                "Docling is not installed. Install it with: pip install docling"
            ) from exc

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure

        if do_ocr:
            ocr_options = self._build_ocr_options(
                engine=ocr_engine,
                force_full_page_ocr=force_full_page_ocr,
            )
            if ocr_options is not None:
                if ocr_lang:
                    ocr_options.lang = ocr_lang
                pipeline_options.ocr_options = ocr_options
            elif ocr_lang:
                pipeline_options.ocr_options.lang = ocr_lang

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.do_ocr = do_ocr
        self.force_full_page_ocr = force_full_page_ocr
        self.ocr_engine = ocr_engine
        self.ocr_lang = ocr_lang or []
        self.do_table_structure = do_table_structure
        self.normalize_pdf = normalize_pdf

    def parse(self, source_path: Path) -> tuple[str, dict[str, Any]]:
        converted, normalized = self._convert(source_path)
        markdown = converted.document.export_to_markdown()
        metadata = {
            "source": str(source_path),
            "engine": self.engine_name,
            "document_name": source_path.name,
            "normalized_pdf": normalized,
            "docling": {
                "ocr": self.do_ocr,
                "force_full_page_ocr": self.force_full_page_ocr,
                "ocr_engine": self.ocr_engine,
                "ocr_lang": self.ocr_lang,
                "table_structure": self.do_table_structure,
            },
        }
        return markdown, metadata

    def _build_ocr_options(
        self,
        engine: str,
        force_full_page_ocr: bool,
    ) -> Any | None:
        if engine == "auto" and not force_full_page_ocr:
            return None

        try:
            import docling.datamodel.pipeline_options as pipeline_options
        except ImportError as exc:
            raise RuntimeError("Docling pipeline options could not be imported.") from exc

        option_names = {
            "auto": "EasyOcrOptions",
            "easyocr": "EasyOcrOptions",
            "tesseract": "TesseractOcrOptions",
            "tesseract_cli": "TesseractCliOcrOptions",
            "rapidocr": "RapidOcrOptions",
        }
        option_name = option_names[engine]
        option_class = getattr(pipeline_options, option_name, None)
        if option_class is None:
            raise RuntimeError(
                f"{option_name} is not available in the installed Docling version."
            )
        return option_class(force_full_page_ocr=force_full_page_ocr)

    def _convert(self, source_path: Path) -> tuple[Any, bool]:
        if self.normalize_pdf:
            with tempfile.TemporaryDirectory(
                prefix="docling_pdf_",
                ignore_cleanup_errors=True,
            ) as temp_dir:
                normalized_path = normalize_pdf(source_path, Path(temp_dir) / "normalized.pdf")
                return self.converter.convert(normalized_path), True

        try:
            return self.converter.convert(source_path), False
        except Exception as exc:
            if "not valid" not in str(exc).lower():
                raise
            with tempfile.TemporaryDirectory(
                prefix="docling_pdf_",
                ignore_cleanup_errors=True,
            ) as temp_dir:
                normalized_path = normalize_pdf(source_path, Path(temp_dir) / "normalized.pdf")
                return self.converter.convert(normalized_path), True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse PDF files with Docling OCR and split Markdown by # headings."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"PDF input directory. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Parsed output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable Docling OCR.",
    )
    parser.add_argument(
        "--force-full-page-ocr",
        action="store_true",
        help="Run OCR over every page. Useful for scanned PDFs, but slower.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["auto", "easyocr", "tesseract", "tesseract_cli", "rapidocr"],
        default="auto",
        help="Docling OCR backend. Default: auto.",
    )
    parser.add_argument(
        "--ocr-lang",
        nargs="+",
        default=None,
        help="OCR language hints, e.g. --ocr-lang ko en.",
    )
    parser.add_argument(
        "--no-table-structure",
        action="store_true",
        help="Disable Docling table structure extraction.",
    )
    parser.add_argument(
        "--no-normalize-pdf",
        action="store_true",
        help="Disable pypdf normalization before Docling conversion.",
    )
    parser.add_argument(
        "--section-min-level",
        type=int,
        default=1,
        help="Minimum Markdown heading level used as a section boundary. Default: 1.",
    )
    parser.add_argument(
        "--section-max-level",
        type=int,
        default=6,
        help="Maximum Markdown heading level used as a section boundary. Default: 6.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-parse files even when output already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Parse only the first N files for quick testing.",
    )
    parser.add_argument(
        "--report-name",
        default="docling_parse_report.csv",
        help="CSV report filename saved under the output directory.",
    )
    return parser.parse_args()


def iter_source_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def normalize_pdf(source_path: Path, output_path: Path) -> Path:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required to normalize PDFs for Docling. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(source_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    with output_path.open("wb") as file:
        writer.write(file)

    return output_path


def make_output_paths(
    source_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    relative_path = source_path.relative_to(input_dir)
    base_dir = output_dir / "docling"
    markdown_path = (base_dir / "markdown" / relative_path).with_suffix(".md")
    metadata_path = (base_dir / "metadata" / relative_path).with_suffix(".json")
    sections_path = (base_dir / "sections_json" / relative_path).with_suffix(".json")
    sections_dir = (base_dir / "sections_md" / relative_path).with_suffix("")
    return markdown_path, metadata_path, sections_path, sections_dir


def split_markdown_by_headings(
    markdown: str,
    source_path: Path,
    min_level: int,
    max_level: int,
) -> list[MarkdownSection]:
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    sections: list[MarkdownSection] = []
    current_title = source_path.stem
    current_level = 1
    current_lines: list[str] = []

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if not content:
            return
        order = len(sections) + 1
        sections.append(
            MarkdownSection(
                section_id=f"{order:04d}_{slugify(current_title)}",
                title=current_title,
                level=current_level,
                order=order,
                content=content,
                source_path=str(source_path),
            )
        )

    for line in markdown.splitlines():
        match = heading_pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            if min_level <= level <= max_level:
                flush()
                current_title = title
                current_level = level
                current_lines = [line]
                continue
        current_lines.append(line)

    flush()
    return sections


def slugify(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^\w가-힣.-]+", "_", value, flags=re.UNICODE).strip("_.")
    slug = re.sub(r"_+", "_", slug)
    return (slug[:max_length].strip("_.") or "section")


def write_sections(sections: list[MarkdownSection], sections_path: Path, sections_dir: Path) -> None:
    sections_path.parent.mkdir(parents=True, exist_ok=True)
    sections_dir.mkdir(parents=True, exist_ok=True)

    sections_path.write_text(
        json.dumps([asdict(section) for section in sections], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    for section in sections:
        section_path = sections_dir / f"{section.section_id}.md"
        section_path.write_text(section.content, encoding="utf-8-sig")


def parse_one(
    parser: DoclingParser,
    source_path: Path,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
    section_min_level: int,
    section_max_level: int,
) -> ParseResult:
    markdown_path, metadata_path, sections_path, sections_dir = make_output_paths(
        source_path,
        input_dir,
        output_dir,
    )

    if markdown_path.exists() and sections_path.exists() and not overwrite:
        return ParseResult(
            source_path=source_path,
            markdown_path=markdown_path,
            metadata_path=metadata_path,
            sections_path=sections_path,
            status="skipped",
            elapsed_seconds=0.0,
            message="output already exists",
        )

    start = time.perf_counter()
    try:
        markdown, metadata = parser.parse(source_path)
        if not markdown.strip():
            raise RuntimeError("Docling produced empty markdown.")

        sections = split_markdown_by_headings(
            markdown=markdown,
            source_path=source_path,
            min_level=section_min_level,
            max_level=section_max_level,
        )
        metadata["section_count"] = len(sections)
        metadata["section_min_level"] = section_min_level
        metadata["section_max_level"] = section_max_level

        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8-sig")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8-sig",
        )
        write_sections(sections, sections_path, sections_dir)

        elapsed = time.perf_counter() - start
        return ParseResult(
            source_path=source_path,
            markdown_path=markdown_path,
            metadata_path=metadata_path,
            sections_path=sections_path,
            status="parsed",
            elapsed_seconds=elapsed,
            text_length=len(markdown),
            section_count=len(sections),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return ParseResult(
            source_path=source_path,
            markdown_path=markdown_path,
            metadata_path=metadata_path,
            sections_path=sections_path,
            status="failed",
            elapsed_seconds=elapsed,
            message=str(exc),
        )


def write_report(results: list[ParseResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "status",
                "source_path",
                "markdown_path",
                "metadata_path",
                "sections_path",
                "text_length",
                "section_count",
                "elapsed_seconds",
                "message",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "status": result.status,
                    "source_path": str(result.source_path),
                    "markdown_path": str(result.markdown_path),
                    "metadata_path": str(result.metadata_path),
                    "sections_path": str(result.sections_path),
                    "text_length": result.text_length,
                    "section_count": result.section_count,
                    "elapsed_seconds": f"{result.elapsed_seconds:.2f}",
                    "message": result.message,
                }
            )


def print_summary(results: list[ParseResult], report_path: Path) -> None:
    parsed = sum(result.status == "parsed" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = sum(result.status == "failed" for result in results)
    print(f"Done: parsed {parsed}, skipped {skipped}, failed {failed}")
    print(f"Report: {report_path}")

    failed_results = [result for result in results if result.status == "failed"]
    if failed_results:
        print("\nFailed files:")
        for result in failed_results[:10]:
            print(f"- {result.source_path}: {result.message}")
        if len(failed_results) > 10:
            print(f"... {len(failed_results) - 10} more")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_path = output_dir / args.report_name

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not 1 <= args.section_min_level <= 6:
        print("--section-min-level must be between 1 and 6.", file=sys.stderr)
        return 1
    if not 1 <= args.section_max_level <= 6:
        print("--section-max-level must be between 1 and 6.", file=sys.stderr)
        return 1
    if args.section_min_level > args.section_max_level:
        print("--section-min-level must be <= --section-max-level.", file=sys.stderr)
        return 1

    source_files = iter_source_files(input_dir)
    if args.limit is not None:
        source_files = source_files[: args.limit]

    if not source_files:
        print(f"No PDF files found in: {input_dir}")
        return 0

    try:
        parser = DoclingParser(
            do_ocr=args.ocr,
            force_full_page_ocr=args.force_full_page_ocr,
            ocr_engine=args.ocr_engine,
            ocr_lang=args.ocr_lang,
            do_table_structure=not args.no_table_structure,
            normalize_pdf=not args.no_normalize_pdf,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results: list[ParseResult] = []
    for index, source_path in enumerate(source_files, start=1):
        print(f"[{index}/{len(source_files)}] docling: {source_path.name}")
        results.append(
            parse_one(
                parser=parser,
                source_path=source_path,
                input_dir=input_dir,
                output_dir=output_dir,
                overwrite=args.overwrite,
                section_min_level=args.section_min_level,
                section_max_level=args.section_max_level,
            )
        )

    write_report(results, report_path)
    print_summary(results, report_path)
    return 0 if not any(result.status == "failed" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
