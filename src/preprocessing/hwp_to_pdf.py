"""
HWP 원본을 PDF로 변환하고, 기존 PDF 원본은 그대로 복사합니다.

이 스크립트는 Windows에 한컴오피스 한글이 설치된 환경을 기준으로 합니다.
HWP 변환은 한컴오피스가 제공하는 HWP COM 객체를 사용하므로 `pywin32`도 필요합니다.

    pip install pywin32

사용 예시:
    python src/preprocessing/hwp_to_pdf.py
    python src/preprocessing/hwp_to_pdf.py --overwrite
    python src/preprocessing/hwp_to_pdf.py --input-dir data/raw/files --output-dir data/raw_pdf
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
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "files"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_pdf"


@dataclass
class ConversionResult:
    source_path: Path
    output_path: Path
    status: str
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "data/raw/files 아래의 .hwp 파일은 PDF로 변환하고, "
            "기존 .pdf 파일은 그대로 data/raw_pdf에 복사합니다. "
            ".docx 등 다른 형식은 무시합니다."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f".hwp와 .pdf 원본 파일이 있는 폴더입니다. 기본값: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"변환/복사된 PDF를 저장할 폴더입니다. 기본값: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 같은 이름의 PDF가 있어도 덮어씁니다.",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="변환 중 한컴오피스 창을 화면에 표시합니다.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="앞에서부터 N개 파일만 처리합니다. 테스트 실행에 유용합니다.",
    )
    parser.add_argument(
        "--report-name",
        default="hwp_to_pdf_report.csv",
        help="출력 폴더에 저장할 CSV 리포트 파일명입니다.",
    )
    return parser.parse_args()


def import_win32com():
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise RuntimeError(
            "HWP 변환에는 pywin32가 필요합니다. 다음 명령으로 설치하세요: pip install pywin32"
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

    # 한컴 버전이 지원하는 경우, 파일 경로 접근 보안 팝업을 줄여줍니다.
    try:
        hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
    except Exception:
        pass

    return pythoncom, hwp


def open_hwp(hwp, source_path: Path) -> bool:
    # 일부 한컴 COM 버전은 선택 인자를 자동으로 채우지 않으므로 명시적으로 넘깁니다.
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
    raise last_error


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

    # SaveAs 대신 액션 API로 PDF 저장을 처리하는 한컴 버전을 위한 대체 경로입니다.
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


def convert_one(hwp, source_path: Path, output_path: Path, overwrite: bool) -> ConversionResult:
    if output_path.exists() and not overwrite:
        return ConversionResult(source_path, output_path, "skipped", "output already exists")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        opened = open_hwp(hwp, source_path)
        if not opened:
            return ConversionResult(source_path, output_path, "failed", "hwp.Open returned False")

        saved = save_as_pdf(hwp, output_path)
        if not saved:
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
        "처리 완료: "
        f"변환 {counts['converted']}개, "
        f"복사 {counts['copied']}개, "
        f"건너뜀 {counts['skipped']}개, "
        f"실패 {counts['failed']}개 "
        f"({elapsed_seconds:.1f}초)"
    )
    print(f"리포트: {report_path}")

    failed = [result for result in results if result.status == "failed"]
    if failed:
        print("\n실패한 파일:")
        for result in failed[:10]:
            print(f"- {result.source_path}: {result.message}")
        if len(failed) > 10:
            print(f"... 외 {len(failed) - 10}개 더 있음")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_path = output_dir / args.report_name

    if not input_dir.exists():
        print(f"입력 폴더가 없습니다: {input_dir}", file=sys.stderr)
        return 1

    source_files = iter_source_files(input_dir)
    if args.limit is not None:
        source_files = source_files[: args.limit]

    if not source_files:
        print(f".hwp 또는 .pdf 파일을 찾지 못했습니다: {input_dir}")
        return 0

    start = time.perf_counter()
    results: list[ConversionResult] = []
    pythoncom = None
    hwp = None

    try:
        hwp_files = [path for path in source_files if path.suffix.lower() == ".hwp"]
        if hwp_files:
            pythoncom, hwp = create_hwp_app(args.visible)

        for index, source_path in enumerate(source_files, start=1):
            output_path = make_output_path(source_path, input_dir, output_dir)
            print(f"[{index}/{len(source_files)}] {source_path.name} -> {output_path.name}")

            if source_path.suffix.lower() == ".pdf":
                results.append(copy_pdf(source_path, output_path, args.overwrite))
            else:
                results.append(convert_one(hwp, source_path, output_path, args.overwrite))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"한컴오피스 한글을 실행하거나 제어하지 못했습니다: {exc}", file=sys.stderr)
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
