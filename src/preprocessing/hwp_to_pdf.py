"""
Convert HWP files to PDF and copy existing PDF files.

Default project flow:
    data/files -> data/pdf -> data/parsed

HWP conversion uses the Hancom HWP COM object. It requires Windows,
Hancom Office/HWP, and pywin32. Existing PDF files can be copied on any OS.

Examples:
    python src/preprocessing/hwp_to_pdf.py
    python src/preprocessing/hwp_to_pdf.py --overwrite
    python src/preprocessing/hwp_to_pdf.py --input-dir data/files --output-dir data/pdf
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "files"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "pdf"


@dataclass
class ConversionResult:
    source_path: Path
    output_path: Path
    status: str
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert .hwp files under data/files to PDF and copy existing .pdf "
            "files into data/pdf. Other formats such as .docx are ignored."
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
        "--overwrite",
        action="store_true",
        help="Overwrite PDFs that already exist.",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show the Hancom Office window during conversion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files. Useful for testing.",
    )
    parser.add_argument(
        "--report-name",
        default="hwp_to_pdf_report.csv",
        help="CSV report filename saved under the output directory.",
    )
    return parser.parse_args()


def import_win32com():
    if sys.platform != "win32":
        raise RuntimeError(
            "HWP conversion requires Windows with Hancom Office/HWP installed. "
            "On a Linux GCP VM, convert HWP to PDF before upload or use a Windows VM."
        )

    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise RuntimeError(
            "HWP conversion requires pywin32. Install it with: pip install pywin32"
        ) from exc
    return pythoncom, win32com.client


def iter_source_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".hwp", ".pdf"}
    )


def make_output_path(source_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = source_path.relative_to(input_dir)
    return (output_dir / relative_path).with_suffix(".pdf")


def create_hwp_app(visible: bool):
    pythoncom, win32_client = import_win32com()
    pythoncom.CoInitialize()
    hwp = win32_client.DispatchEx("HWPFrame.HwpObject")

    try:
        hwp.XHwpWindows.Item(0).Visible = visible
    except Exception:
        pass

    try:
        hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
    except Exception:
        pass

    return pythoncom, hwp


def open_hwp(hwp, source_path: Path) -> bool:
    open_args = [
        (str(source_path), "HWP", "forceopen:true"),
        (str(source_path), "HWP", ""),
        (str(source_path), "", ""),
    ]
    last_error = None
    for args in open_args:
        try:
            return bool(hwp.Open(*args))
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return False


def save_as_pdf(hwp, output_path: Path) -> bool:
    save_args = [
        (str(output_path), "PDF", ""),
        (str(output_path), "pdf", ""),
        (str(output_path), "PDF", "lock:false"),
    ]
    last_error = None
    for args in save_args:
        try:
            if hwp.SaveAs(*args):
                return True
        except Exception as exc:
            last_error = exc

    try:
        parameter_set = hwp.HParameterSet.HFileOpenSave
        hwp.HAction.GetDefault("FileSaveAsPdf", parameter_set.HSet)
        parameter_set.filename = str(output_path)
        parameter_set.Format = "PDF"
        return bool(hwp.HAction.Execute("FileSaveAsPdf", parameter_set.HSet))
    except Exception as exc:
        last_error = exc

    if last_error is not None:
        raise last_error
    return False


def close_hwp_app(pythoncom, hwp) -> None:
    try:
        hwp.Quit()
    finally:
        pythoncom.CoUninitialize()


def convert_hwp_one(
    hwp,
    source_path: Path,
    output_path: Path,
    overwrite: bool,
) -> ConversionResult:
    if output_path.exists() and not overwrite:
        return ConversionResult(source_path, output_path, "skipped", "output already exists")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if not open_hwp(hwp, source_path):
            return ConversionResult(source_path, output_path, "failed", "hwp.Open returned False")
        if not save_as_pdf(hwp, output_path):
            return ConversionResult(source_path, output_path, "failed", "hwp.SaveAs returned False")
        return ConversionResult(source_path, output_path, "converted")
    except Exception as exc:
        return ConversionResult(source_path, output_path, "failed", str(exc))
    finally:
        try:
            hwp.Clear(1)
        except Exception:
            pass


def copy_pdf(source_path: Path, output_path: Path, overwrite: bool) -> ConversionResult:
    if output_path.exists() and not overwrite:
        return ConversionResult(source_path, output_path, "skipped", "output already exists")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(source_path, output_path)
        return ConversionResult(source_path, output_path, "copied")
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
    pythoncom = None
    hwp = None

    try:
        pending_files: list[tuple[Path, Path]] = []
        for source_path in source_files:
            output_path = make_output_path(source_path, input_dir, output_dir)
            if output_path.exists() and not args.overwrite:
                results.append(
                    ConversionResult(source_path, output_path, "skipped", "output already exists")
                )
            else:
                pending_files.append((source_path, output_path))

        hwp_files = [
            source_path
            for source_path, _ in pending_files
            if source_path.suffix.lower() == ".hwp"
        ]
        if hwp_files:
            pythoncom, hwp = create_hwp_app(args.visible)

        for index, (source_path, output_path) in enumerate(pending_files, start=1):
            print(f"[{index}/{len(pending_files)}] {source_path.name} -> {output_path.name}")

            if source_path.suffix.lower() == ".pdf":
                results.append(copy_pdf(source_path, output_path, args.overwrite))
            else:
                results.append(convert_hwp_one(hwp, source_path, output_path, args.overwrite))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed to control Hancom Office/HWP: {exc}", file=sys.stderr)
        return 1
    finally:
        if pythoncom is not None and hwp is not None:
            close_hwp_app(pythoncom, hwp)

    elapsed = time.perf_counter() - start
    write_report(results, report_path)
    print_summary(results, elapsed, report_path)
    return 0 if not any(result.status == "failed" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
