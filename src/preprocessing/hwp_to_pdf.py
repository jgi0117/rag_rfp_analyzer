"""
Convert HWP files to PDF and copy existing PDF files.

Default project flow:
    data/files -> data/pdf -> data/parsed

On Windows, HWP conversion uses the Hancom HWP COM object. On Linux, this
script tries LibreOffice headless first, then an optional hwp5odt -> LibreOffice
fallback if hwp5odt is installed. Existing PDF files are copied as-is.

Examples:
    python src/preprocessing/hwp_to_pdf.py
    python src/preprocessing/hwp_to_pdf.py --overwrite
    python src/preprocessing/hwp_to_pdf.py --input-dir data/files --output-dir data/pdf
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
        help="Show the Hancom Office window during Windows COM conversion.",
    )
    parser.add_argument(
        "--converter",
        choices=["auto", "win32", "libreoffice", "hwp5odt"],
        default="auto",
        help=(
            "HWP conversion backend. auto uses win32 on Windows and LibreOffice "
            "on Linux/macOS. Default: auto."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Seconds to wait for each external conversion command. Default: 180.",
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
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise RuntimeError(
            "HWP conversion requires pywin32. Install it with: pip install pywin32"
        ) from exc
    return pythoncom, win32com.client


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
        if path.is_file() and path.suffix.lower() in {".hwp", ".pdf"}
    )


def make_output_path(source_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = source_path.relative_to(input_dir)
    return (output_dir / relative_path).with_suffix(".pdf")


def create_hwp_app(visible: bool):
    if sys.platform != "win32":
        raise RuntimeError(
            "The win32 converter requires Windows with Hancom Office/HWP installed. "
            "Use --converter libreoffice or --converter hwp5odt on Linux."
        )

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


def convert_hwp_with_win32(
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
    with tempfile.TemporaryDirectory(prefix="hwp_to_pdf_") as temp_dir:
        temp_output_dir = Path(temp_dir)
        command = [
            libreoffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(temp_output_dir),
            str(source_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        produced_pdf = temp_output_dir / f"{source_path.stem}.pdf"
        if result.returncode != 0 or not produced_pdf.exists():
            message = (
                f"LibreOffice conversion failed with exit code {result.returncode}. "
                f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
            )
            return ConversionResult(source_path, output_path, "failed", message)

        shutil.move(str(produced_pdf), output_path)
        return ConversionResult(source_path, output_path, "converted", "libreoffice")


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
            "hwp5odt executable not found. Install pyhwp/hwp5 tools if LibreOffice cannot read HWP.",
        )

    libreoffice = find_executable(["soffice", "libreoffice"])
    if libreoffice is None:
        return ConversionResult(
            source_path,
            output_path,
            "failed",
            "LibreOffice executable not found. hwp5odt fallback still needs LibreOffice for ODT -> PDF.",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hwp5odt_to_pdf_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        odt_path = temp_dir_path / f"{source_path.stem}.odt"

        odt_result = run_hwp5odt(hwp5odt, source_path, odt_path, timeout)
        if odt_result is not None:
            return ConversionResult(source_path, output_path, "failed", odt_result)

        command = [
            libreoffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(temp_dir_path),
            str(odt_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        produced_pdf = temp_dir_path / f"{odt_path.stem}.pdf"
        if result.returncode != 0 or not produced_pdf.exists():
            message = (
                f"ODT -> PDF conversion failed with exit code {result.returncode}. "
                f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
            )
            return ConversionResult(source_path, output_path, "failed", message)

        shutil.move(str(produced_pdf), output_path)
        return ConversionResult(source_path, output_path, "converted", "hwp5odt+libreoffice")


def run_hwp5odt(hwp5odt: str, source_path: Path, odt_path: Path, timeout: int) -> str | None:
    command_variants = [
        [hwp5odt, "--output", str(odt_path), str(source_path)],
        [hwp5odt, "-o", str(odt_path), str(source_path)],
    ]
    errors: list[str] = []
    for command in command_variants:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0 and odt_path.exists():
            return None
        errors.append(
            f"{' '.join(command[:2])}: exit={result.returncode}, "
            f"stdout={result.stdout.strip()}, stderr={result.stderr.strip()}"
        )

    result = subprocess.run(
        [hwp5odt, str(source_path)],
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode == 0 and result.stdout:
        odt_path.write_bytes(result.stdout)
        return None
    errors.append(
        f"{hwp5odt} <source>: exit={result.returncode}, stderr={result.stderr.decode(errors='ignore').strip()}"
    )
    return "; ".join(errors)


def convert_hwp_one(
    hwp,
    source_path: Path,
    output_path: Path,
    overwrite: bool,
    converter: str,
    timeout: int,
) -> ConversionResult:
    selected_converter = converter
    if converter == "auto":
        selected_converter = "win32" if sys.platform == "win32" else "libreoffice"

    try:
        if selected_converter == "win32":
            return convert_hwp_with_win32(hwp, source_path, output_path, overwrite)
        if selected_converter == "libreoffice":
            result = convert_hwp_with_libreoffice(source_path, output_path, overwrite, timeout)
            if result.status == "converted" or converter != "auto":
                return result
            fallback = convert_hwp_with_hwp5odt(source_path, output_path, overwrite, timeout)
            if fallback.status == "converted":
                return fallback
            result.message = f"{result.message}; hwp5odt fallback: {fallback.message}"
            return result
        if selected_converter == "hwp5odt":
            return convert_hwp_with_hwp5odt(source_path, output_path, overwrite, timeout)
    except subprocess.TimeoutExpired as exc:
        return ConversionResult(
            source_path,
            output_path,
            "failed",
            f"Conversion timed out after {exc.timeout} seconds.",
        )
    except Exception as exc:
        return ConversionResult(source_path, output_path, "failed", str(exc))

    return ConversionResult(
        source_path,
        output_path,
        "failed",
        f"Unsupported converter: {converter}",
    )


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

        needs_win32 = args.converter == "win32" or (
            args.converter == "auto" and sys.platform == "win32"
        )
        hwp_files = [
            source_path
            for source_path, _ in pending_files
            if source_path.suffix.lower() == ".hwp"
        ]
        if hwp_files and needs_win32:
            pythoncom, hwp = create_hwp_app(args.visible)

        for index, (source_path, output_path) in enumerate(pending_files, start=1):
            print(f"[{index}/{len(pending_files)}] {source_path.name} -> {output_path.name}")

            if source_path.suffix.lower() == ".pdf":
                results.append(copy_pdf(source_path, output_path, args.overwrite))
            else:
                results.append(
                    convert_hwp_one(
                        hwp=hwp,
                        source_path=source_path,
                        output_path=output_path,
                        overwrite=args.overwrite,
                        converter=args.converter,
                        timeout=args.timeout,
                    )
                )
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
