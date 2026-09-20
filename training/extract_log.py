import argparse
import csv
import re
from pathlib import Path


ANSI_PATTERN = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)

# Ví dụ:
# togrow3 summary: 260 layers, 3,011,970 parameters,
# 3,011,970 gradients, 9.1 GFLOPs
#
# Không khớp với "summary (fused)"
MODEL_SUMMARY_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<model_name>[^\r\n]+?)\s+summary:\s*
    (?P<layers>[\d,]+)\s+layers,\s*
    (?P<parameters>[\d,]+)\s+parameters,\s*
    (?P<gradients>[\d,]+)\s+gradients,\s*
    (?P<gflops>[\d.]+)\s+GFLOPs
    \s*$
    """,
    re.VERBOSE,
)

VALIDATING_PATTERN = re.compile(
    r"""
    ^\s*Validating\s+
    (?P<checkpoint>.+?\.pt)
    \.{0,3}\s*$
    """,
    re.VERBOSE,
)

METRICS_PATTERN = re.compile(
    r"""
    ^\s*all\s+
    (?P<images>\d+)\s+
    (?P<instances>\d+)\s+
    (?P<precision>[\d.eE+-]+)\s+
    (?P<recall>[\d.eE+-]+)\s+
    (?P<map50>[\d.eE+-]+)\s+
    (?P<map50_95>[\d.eE+-]+)
    \s*$
    """,
    re.VERBOSE,
)

SPEED_PATTERN = re.compile(
    r"""
    ^\s*Speed:\s*
    (?P<preprocess>[\d.]+)ms\s+preprocess,\s*
    (?P<inference>[\d.]+)ms\s+inference,\s*
    (?P<loss>[\d.]+)ms\s+loss,\s*
    (?P<postprocess>[\d.]+)ms\s+postprocess
    """,
    re.VERBOSE,
)

RESULTS_SAVED_PATTERN = re.compile(
    r"^\s*Results saved to\s+"
)


def clean_log(text: str) -> str:
    text = ANSI_PATTERN.sub("", text)
    text = text.replace("\r", "\n")
    return text


def get_run_name(checkpoint_path: str) -> str:
    path = Path(checkpoint_path)

    if path.parent.name == "weights":
        return path.parent.parent.name

    return path.stem


def parse_number(value: str) -> int:
    return int(value.replace(",", ""))


def extract_results(log_path: Path) -> list[dict]:
    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = clean_log(text).splitlines()

    results = []

    # Summary gần nhất của lần train hiện tại
    latest_summary = None

    current_checkpoint = None
    current_summary = None
    current_metrics = None
    current_speed = None

    for line in lines:
        # Lấy summary gốc trước khi train
        summary_match = MODEL_SUMMARY_PATTERN.match(line)

        if summary_match:
            summary = summary_match.groupdict()

            latest_summary = {
                "model_name": summary["model_name"].strip(),
                "layers": parse_number(summary["layers"]),
                "parameters": parse_number(
                    summary["parameters"]
                ),
                "gradients": parse_number(
                    summary["gradients"]
                ),
                "gflops": float(summary["gflops"]),
            }

            continue

        # Bắt đầu validation cuối cùng
        validating_match = VALIDATING_PATTERN.match(line)

        if validating_match:
            current_checkpoint = (
                validating_match.group("checkpoint").strip()
            )

            # Sao chép summary của đúng lần train vừa xong
            current_summary = (
                latest_summary.copy()
                if latest_summary is not None
                else None
            )

            current_metrics = None
            current_speed = None
            continue

        if current_checkpoint is None:
            continue

        metrics_match = METRICS_PATTERN.match(line)

        if metrics_match:
            current_metrics = metrics_match.groupdict()
            continue

        speed_match = SPEED_PATTERN.match(line)

        if speed_match:
            current_speed = speed_match.groupdict()
            continue

        if RESULTS_SAVED_PATTERN.match(line):
            if current_metrics is None:
                print(
                    "[Cảnh báo] Không tìm thấy metrics cho: "
                    f"{current_checkpoint}"
                )
            else:
                row = make_row(
                    checkpoint=current_checkpoint,
                    summary=current_summary,
                    metrics=current_metrics,
                    speed=current_speed,
                )

                results.append(row)

            current_checkpoint = None
            current_summary = None
            current_metrics = None
            current_speed = None

    # Trường hợp cuối file không có "Results saved to"
    if (
        current_checkpoint is not None
        and current_metrics is not None
    ):
        results.append(
            make_row(
                checkpoint=current_checkpoint,
                summary=current_summary,
                metrics=current_metrics,
                speed=current_speed,
            )
        )

    return results


def make_row(
    checkpoint: str,
    summary: dict | None,
    metrics: dict,
    speed: dict | None,
) -> dict:
    row = {
        "run_name": get_run_name(checkpoint),
        "model_name": "",
        "checkpoint_path": checkpoint,
        "layers": "",
        "parameters": "",
        "gradients": "",
        "gflops": "",
        "images": int(metrics["images"]),
        "instances": int(metrics["instances"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "map50": float(metrics["map50"]),
        "map50_95": float(metrics["map50_95"]),
        "preprocess_ms": "",
        "inference_ms": "",
        "postprocess_ms": "",
    }

    if summary is not None:
        row["model_name"] = summary["model_name"]
        row["layers"] = summary["layers"]
        row["parameters"] = summary["parameters"]
        row["gradients"] = summary["gradients"]
        row["gflops"] = summary["gflops"]
    else:
        print(
            "[Cảnh báo] Không tìm thấy summary gốc cho: "
            f"{checkpoint}"
        )

    if speed is not None:
        row["preprocess_ms"] = float(speed["preprocess"])
        row["inference_ms"] = float(speed["inference"])
        row["postprocess_ms"] = float(
            speed["postprocess"]
        )

    return row


def save_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        raise RuntimeError(
            "Không trích xuất được kết quả nào."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "run_name",
        "model_name",
        "checkpoint_path",
        "layers",
        "parameters",
        "gradients",
        "gflops",
        "images",
        "instances",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trích xuất summary gốc và kết quả validation "
            "của nhiều lần train YOLO."
        )
    )

    parser.add_argument(
        "log_file",
        type=Path,
        help="File log đầu vào.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("model_results.csv"),
        help="File CSV đầu ra.",
    )

    args = parser.parse_args()

    rows = extract_results(args.log_file)
    save_csv(rows, args.output)

    print(f"\nĐã trích xuất {len(rows)} mô hình:\n")

    for row in rows:
        print(
            f"{row['run_name']}: "
            f"model={row['model_name']}, "
            f"layers={row['layers']}, "
            f"parameters={row['parameters']}, "
            f"GFLOPs={row['gflops']}, "
            f"P={row['precision']}, "
            f"R={row['recall']}, "
            f"mAP50={row['map50']}, "
            f"mAP50-95={row['map50_95']}"
        )

    print(f"\nĐã lưu: {args.output.resolve()}")


if __name__ == "__main__":
    main()