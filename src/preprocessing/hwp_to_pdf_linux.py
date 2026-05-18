"""
Convert HWP files to PDF on Linux and copy existing PDF files.

Default project flow:
    data/raw/files -> data/raw_pdf -> data/parsed

This Linux-only script uses LibreOffice headless first. If LibreOffice cannot
read a HWP file directly, --converter auto falls back to hwp5odt -> LibreOffice.

VM setup examples:
    sudo apt-get update
    sudo apt-get install -y libreoffice libreoffice-hwpfilter fonts-nanum
    pip install pyhwp2

Usage examples:
    python src/preprocessing/hwp_to_pdf_linux.py
    python src/preprocessing/hwp_to_pdf_linux.py --overwrite
    python src/preprocessing/hwp_to_pdf_linux.py --converter hwp5odt --limit 3
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "files"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_pdf"
SUPPORTED_EXTENSIONS = {".hwp", ".pdf"}


@dataclass
class ConversionResult:
    source_path: Path
    output_path: Path
    status: str
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Linux-only converter for .hwp files under data/raw/files. Existing "
            ".pdf files are copied into data/raw_pdf."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing source .hwp and .pdf files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for converted/copied PDFs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--converter",
        choices=["auto", "libreoffice", "hwp5odt"],
        default="auto",
        help=(
            "HWP conversion backend. auto tries LibreOffice first, then hwp5odt. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite PDFs that already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="Seconds to wait for each external conversion command. Default: 240.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files. Useful for testing.",
    )
    parser.add_argument(
        "--report-name",
        default="hwp_to_pdf_linux_report.csv",
        help="CSV report filename saved under the output directory.",
    )
    return parser.parse_args()


def find_executable(candidates: list[str]) -> str | None:
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            return executable
    return None


def iter_source_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def safe_path_part(value: str) -> str:
    safe = "".join("_" if char in '<>:"/\\|?*' else char for char in value)
    safe = safe.strip(" .")
    return safe or "unnamed"


def make_output_path(source_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = Path(
        *[safe_path_part(part) for part in source_path.relative_to(input_dir).parts]
    )
    return output_dir / relative_path.parent / f"{safe_path_part(relative_path.stem)}.pdf"


def copy_pdf(source_path: Path, output_path: Path, overwrite: bool) -> ConversionResult:
    if output_path.exists() and not overwrite:
        return ConversionResult(source_path, output_path, "skipped", "output already exists")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source_path, output_path)
        return ConversionResult(source_path, output_path, "copied")
    except Exception as exc:
        return ConversionResult(source_path, output_path, "failed", str(exc))


def run_command(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def convert_hwp_with_libreoffice(
    source_path: Path,
    output_path: Path,
    overwrite: bool,
    timeout: int,
) -> ConversionResult:
    if output_path.exists() and not overwrite:
        return ConversionResult(source_path, output_path, "skipped", "output already exists")

    libreoffice = find_executable(["soffice", "libreoffice"])
    if libreoffice is None:
        return ConversionResult(
            source_path,
            output_path,
            "failed",
            "LibreOffice executable not found. Install libreoffice on the VM.",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hwp_to_pdf_linux_") as temp_dir:
        temp_output_dir = Path(temp_dir)
        command = [
            libreoffice,
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--convert-to",
            "pdf",
            "--outdir",
            str(temp_output_dir),
            str(source_path),
        ]
        result = run_command(command, timeout)
        produced_pdf = temp_output_dir / f"{source_path.stem}.pdf"

        if result.returncode != 0 or not produced_pdf.exists():
            message = (
                f"LibreOffice conversion failed with exit code {result.returncode}. "
                f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
            )
            return ConversionResult(source_path, output_path, "failed", message)

        shutil.move(str(produced_pdf), output_path)
        return ConversionResult(source_path, output_path, "converted", "libreoffice")


def run_hwp5odt(hwp5odt: str, source_path: Path, odt_path: Path, timeout: int) -> str | None:
    command_variants = [
        [hwp5odt, "--output", str(odt_path), str(source_path)],
        [hwp5odt, "-o", str(odt_path), str(source_path)],
    ]
    errors: list[str] = []
    for command in command_variants:
        result = run_command(command, timeout)
        if result.returncode == 0 and odt_path.exists():
            return None
        errors.append(
            f"{' '.join(command[:2])}: exit={result.returncode}, "
            f"stdout={result.stdout.strip()}, stderr={result.stderr.strip()}"
        )

    raw_result = subprocess.run(
        [hwp5odt, str(source_path)],
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if raw_result.returncode == 0 and raw_result.stdout:
        odt_path.write_bytes(raw_result.stdout)
        return None
    errors.append(
        f"{hwp5odt} <source>: exit={raw_result.returncode}, "
        f"stderr={raw_result.stderr.decode(errors='ignore').strip()}"
    )
    return "; ".join(errors)


def convert_hwp_with_hwp5odt(
    source_path: Path,
    output_path: Path,
    overwrite: bool,
    timeout: int,
) -> ConversionResult:
    if output_path.exists() and not overwrite:
        return ConversionResult(source_path, output_path, "skipped", "output already exists")

    hwp5odt = find_executable(["hwp5odt"])
    if hwp5odt is None:
        return ConversionResult(
            source_path,
            output_path,
            "failed",
            "hwp5odt executable not found. Install pyhwp/pyhwp2 tools on the VM.",
        )

    libreoffice = find_executable(["soffice", "libreoffice"])
    if libreoffice is None:
        return ConversionResult(
            source_path,
            output_path,
            "failed",
            "LibreOffice executable not found. hwp5odt still needs LibreOffice for ODT -> PDF.",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hwp5odt_to_pdf_linux_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        odt_path = temp_dir_path / f"{source_path.stem}.odt"

        odt_error = run_hwp5odt(hwp5odt, source_path, odt_path, timeout)
        if odt_error is not None:
            return ConversionResult(source_path, output_path, "failed", odt_error)

        command = [
            libreoffice,
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--convert-to",
            "pdf",
            "--outdir",
            str(temp_dir_path),
            str(odt_path),
        ]
        result = run_command(command, timeout)
        produced_pdf = temp_dir_path / f"{odt_path.stem}.pdf"

        if result.returncode != 0 or not produced_pdf.exists():
            message = (
                f"ODT -> PDF conversion failed with exit code {result.returncode}. "
                f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
            )
            return ConversionResult(source_path, output_path, "failed", message)

        shutil.move(str(produced_pdf), output_path)
        return ConversionResult(source_path, output_path, "converted", "hwp5odt+libreoffice")


def convert_hwp_one(
    source_path: Path,
    output_path: Path,
    overwrite: bool,
    converter: str,
    timeout: int,
) -> ConversionResult:
    try:
        if converter == "libreoffice":
            return convert_hwp_with_libreoffice(source_path, output_path, overwrite, timeout)
        if converter == "hwp5odt":
            return convert_hwp_with_hwp5odt(source_path, output_path, overwrite, timeout)

        libreoffice_result = convert_hwp_with_libreoffice(
            source_path,
            output_path,
            overwrite,
            timeout,
        )
        if libreoffice_result.status in {"converted", "skipped"}:
            return libreoffice_result

        hwp5odt_result = convert_hwp_with_hwp5odt(
            source_path,
            output_path,
            overwrite,
            timeout,
        )
        if hwp5odt_result.status == "converted":
            return hwp5odt_result

        libreoffice_result.message = (
            f"{libreoffice_result.message}; hwp5odt fallback: {hwp5odt_result.message}"
        )
        return libreoffice_result
    except subprocess.TimeoutExpired as exc:
        return ConversionResult(
            source_path,
            output_path,
            "failed",
            f"Conversion timed out after {exc.timeout} seconds.",
        )
    except Exception as exc:
        return ConversionResult(source_path, output_path, "failed", str(exc))


def write_report(results: list[ConversionResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["status", "source_path", "output_path", "message"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "status": result.status,
                    "source_path": str(result.source_path),
                    "output_path": str(result.output_path),
                    "message": result.message,
                }
            )


def print_summary(results: list[ConversionResult], elapsed_seconds: float, report_path: Path) -> None:
    counts = {
        status: sum(result.status == status for result in results)
        for status in ["converted", "copied", "skipped", "failed"]
    }
    print(
        "Done: "
        f"converted {counts['converted']}, "
        f"copied {counts['copied']}, "
        f"skipped {counts['skipped']}, "
        f"failed {counts['failed']} "
        f"({elapsed_seconds:.1f}s)"
    )
    print(f"Report: {report_path}")

    failed = [result for result in results if result.status == "failed"]
    if failed:
        print("\nFailed files:")
        for result in failed[:10]:
            print(f"- {result.source_path}: {result.message}")
        if len(failed) > 10:
            print(f"... {len(failed) - 10} more")


def main() -> int:
    args = parse_args()

    if sys.platform.startswith("win"):
        print(
            "This script is Linux-oriented and does not use Hancom COM. "
            "Use hwp_to_pdf.py on Windows.",
            file=sys.stderr,
        )
        return 1

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_path = output_dir / args.report_name

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    source_files = iter_source_files(input_dir)
    if args.limit is not None:
        source_files = source_files[: args.limit]

    if not source_files:
        print(f"No .hwp or .pdf files found in: {input_dir}")
        return 0

    start = time.perf_counter()
    results: list[ConversionResult] = []
    pending_files: list[tuple[Path, Path]] = []

    for source_path in source_files:
        output_path = make_output_path(source_path, input_dir, output_dir)
        if output_path.exists() and not args.overwrite:
            results.append(
                ConversionResult(source_path, output_path, "skipped", "output already exists")
            )
        else:
            pending_files.append((source_path, output_path))

    for index, (source_path, output_path) in enumerate(pending_files, start=1):
        print(f"[{index}/{len(pending_files)}] {source_path.name} -> {output_path.name}", flush=True)

        if source_path.suffix.lower() == ".pdf":
            result = copy_pdf(source_path, output_path, args.overwrite)
        else:
            result = convert_hwp_one(
                source_path=source_path,
                output_path=output_path,
                overwrite=args.overwrite,
                converter=args.converter,
                timeout=args.timeout,
            )
        results.append(result)

    elapsed = time.perf_counter() - start
    write_report(results, report_path)
    print_summary(results, elapsed, report_path)
    return 0 if not any(result.status == "failed" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
