"""
Parse PDF documents with Docling and split Markdown by headings.

Examples:
    python src/preprocessing/document_parser.py --limit 3
    python src/preprocessing/document_parser.py --ocr --limit 3
    python src/preprocessing/document_parser.py --ocr --force-full-page-ocr --overwrite
    python src/preprocessing/document_parser.py --ocr --page-batch-size 5 --num-threads 1
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used when tqdm is not installed.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_pdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "parsed"
SUPPORTED_EXTENSIONS = {".pdf"}
DEFAULT_PAGE_BATCH_SIZE = 5
DEFAULT_NUM_THREADS = 1
DEFAULT_DOCUMENT_TIMEOUT = 120.0
T = TypeVar("T")


def progress_bar(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str,
    unit: str,
    leave: bool = True,
) -> Iterable[T]:
    if tqdm is None or not sys.stderr.isatty():
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave)


def progress_write(message: str) -> None:
    if tqdm is None or not sys.stderr.isatty():
        print(message, flush=True)
        return
    tqdm.write(message)


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
        num_threads: int,
        document_timeout: float | None,
        device: str,
    ) -> None:
        try:
            from docling.datamodel.accelerator_options import AcceleratorOptions
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
        pipeline_options.document_timeout = document_timeout
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=device,
        )

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
        self.num_threads = num_threads
        self.document_timeout = document_timeout
        self.device = device

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
        del converted
        gc.collect()
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
        help="Disable Docling table structure extraction. This is the default unless --table-structure is set.",
    )
    parser.add_argument(
        "--table-structure",
        action="store_true",
        help="Enable Docling table structure extraction. Slower and may require model downloads.",
    )
    parser.add_argument(
        "--no-normalize-pdf",
        action="store_true",
        help="Disable pypdf normalization before Docling conversion.",
    )
    parser.add_argument(
        "--page-batch-size",
        type=int,
        default=DEFAULT_PAGE_BATCH_SIZE,
        help=(
            "Split each PDF into N-page temporary PDFs before Docling parsing to reduce "
            "peak memory and show progress on large files. Use 0 to parse each file at once. "
            f"Default: {DEFAULT_PAGE_BATCH_SIZE}."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=(
            "Limit common native thread pools before Docling loads. Lower values usually "
            f"reduce memory. Default: {DEFAULT_NUM_THREADS}."
        ),
    )
    parser.add_argument(
        "--document-timeout",
        type=float,
        default=DEFAULT_DOCUMENT_TIMEOUT,
        help=(
            "Docling timeout per temporary PDF in seconds. Use 0 to disable. "
            f"Default: {DEFAULT_DOCUMENT_TIMEOUT}."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "auto", "cuda", "mps", "xpu"],
        default="cpu",
        help="Docling inference device. CPU is the safest default. Default: cpu.",
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


def configure_memory_limits(num_threads: int) -> None:
    if num_threads < 1:
        raise ValueError("--num-threads must be >= 1.")

    thread_count = str(num_threads)
    os.environ.setdefault("OMP_NUM_THREADS", thread_count)
    os.environ.setdefault("MKL_NUM_THREADS", thread_count)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", thread_count)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", thread_count)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


def count_pdf_pages(source_path: Path) -> int:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to count PDF pages. Install it with: pip install pypdf") from exc

    return len(PdfReader(str(source_path)).pages)


def split_pdf_batches(source_path: Path, batch_size: int, temp_dir: Path) -> list[tuple[Path, int, int]]:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise RuntimeError("pypdf is required to split PDF pages. Install it with: pip install pypdf") from exc

    reader = PdfReader(str(source_path))
    page_count = len(reader.pages)
    if batch_size <= 0 or page_count <= batch_size:
        return [(source_path, 1, page_count)]

    batches: list[tuple[Path, int, int]] = []
    for start_index in range(0, page_count, batch_size):
        end_index = min(start_index + batch_size, page_count)
        writer = PdfWriter()
        for page_index in range(start_index, end_index):
            writer.add_page(reader.pages[page_index])

        batch_path = temp_dir / f"{source_path.stem}_pages_{start_index + 1}_{end_index}.pdf"
        with batch_path.open("wb") as file:
            writer.write(file)
        batches.append((batch_path, start_index + 1, end_index))
        del writer

    del reader
    gc.collect()
    return batches


def parse_markdown_in_page_batches(
    parser: Any,
    source_path: Path,
    page_batch_size: int,
) -> tuple[str, dict[str, Any]]:
    if page_batch_size <= 0:
        return parser.parse(source_path)

    page_count = count_pdf_pages(source_path)
    if page_count <= page_batch_size:
        return parser.parse(source_path)

    batch_markdowns: list[str] = []
    batch_metadata: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="docling_page_batches_",
        ignore_cleanup_errors=True,
    ) as temp_dir:
        batches = split_pdf_batches(source_path, page_batch_size, Path(temp_dir))
        batch_iter = progress_bar(
            batches,
            total=len(batches),
            desc=f"pages {source_path.name}",
            unit="batch",
            leave=False,
        )
        for batch_index, (batch_path, start_page, end_page) in enumerate(batch_iter, start=1):
            progress_write(
                f"  - batch {batch_index}/{len(batches)}: pages {start_page}-{end_page}"
            )
            markdown, metadata = parser.parse(batch_path)
            batch_markdowns.append(
                f"<!-- pages: {start_page}-{end_page} -->\n\n{markdown.strip()}"
            )
            batch_metadata.append(
                {
                    "batch": batch_index,
                    "start_page": start_page,
                    "end_page": end_page,
                    "temp_source": str(batch_path),
                    "metadata": metadata,
                    "text_length": len(markdown),
                }
            )
            del markdown
            gc.collect()

    metadata = {
        "source": str(source_path),
        "engine": parser.engine_name,
        "document_name": source_path.name,
        "page_count": page_count,
        "page_batch_size": page_batch_size,
        "batch_count": len(batch_metadata),
        "batches": batch_metadata,
    }
    return "\n\n".join(batch_markdowns).strip(), metadata


def safe_path_part(value: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", value).strip(" .")
    return safe or "unnamed"


def make_output_paths(
    source_path: Path,
    input_dir: Path,
    output_dir: Path,
    engine_name: str,
) -> tuple[Path, Path, Path, Path]:
    relative_path = Path(
        *[safe_path_part(part) for part in source_path.relative_to(input_dir).parts]
    )
    relative_parent = relative_path.parent
    output_stem = safe_path_part(relative_path.stem)
    base_dir = output_dir / engine_name
    markdown_path = base_dir / "markdown" / relative_parent / f"{output_stem}.md"
    metadata_path = base_dir / "metadata" / relative_parent / f"{output_stem}.json"
    sections_path = base_dir / "sections_json" / relative_parent / f"{output_stem}.json"
    sections_dir = base_dir / "sections_md" / relative_parent / output_stem
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
    parser: Any,
    source_path: Path,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
    section_min_level: int,
    section_max_level: int,
    page_batch_size: int,
) -> ParseResult:
    markdown_path, metadata_path, sections_path, sections_dir = make_output_paths(
        source_path,
        input_dir,
        output_dir,
        parser.engine_name,
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
        markdown, metadata = parse_markdown_in_page_batches(
            parser=parser,
            source_path=source_path,
            page_batch_size=page_batch_size,
        )
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
        gc.collect()

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
    try:
        configure_memory_limits(args.num_threads)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
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
    if args.page_batch_size < 0:
        print("--page-batch-size must be >= 0.", file=sys.stderr)
        return 1
    if args.document_timeout < 0:
        print("--document-timeout must be >= 0.", file=sys.stderr)
        return 1

    source_files = iter_source_files(input_dir)
    if args.limit is not None:
        source_files = source_files[: args.limit]

    if not source_files:
        print(f"No PDF files found in: {input_dir}")
        return 0

    document_timeout = None if args.document_timeout == 0 else args.document_timeout
    try:
        table_structure = args.table_structure and not args.no_table_structure
        progress_write(
            "Initializing Docling "
            f"(ocr={args.ocr}, tables={table_structure}, timeout={document_timeout}, "
            f"threads={args.num_threads}, device={args.device})"
        )
        parser = DoclingParser(
            do_ocr=args.ocr,
            force_full_page_ocr=args.force_full_page_ocr,
            ocr_engine=args.ocr_engine,
            ocr_lang=args.ocr_lang,
            do_table_structure=table_structure,
            normalize_pdf=not args.no_normalize_pdf,
            num_threads=args.num_threads,
            document_timeout=document_timeout,
            device=args.device,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results: list[ParseResult] = []
    file_iter = progress_bar(
        source_files,
        total=len(source_files),
        desc=f"{parser.engine_name} files",
        unit="file",
    )
    for index, source_path in enumerate(file_iter, start=1):
        progress_write(f"[{index}/{len(source_files)}] {parser.engine_name}: {source_path.name}")
        try:
            results.append(
                parse_one(
                    parser=parser,
                    source_path=source_path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    overwrite=args.overwrite,
                    section_min_level=args.section_min_level,
                    section_max_level=args.section_max_level,
                    page_batch_size=args.page_batch_size,
                )
            )
        except KeyboardInterrupt:
            progress_write("Interrupted. Writing partial report before exit.")
            write_report(results, report_path)
            raise

    write_report(results, report_path)
    print_summary(results, report_path)
    return 0 if not any(result.status == "failed" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
