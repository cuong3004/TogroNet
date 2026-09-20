"""Benchmark cac file .tflite (dung luong, RAM peak, do tre, FPS) TREN CHINH
Raspberry Pi -- khong can dataset that, chi can 1 anh dummy dung shape/dtype
input cua tung model. Chay bang tflite-runtime (nhe) hoac ai-edge-litert;
neu khong co ca hai thi fallback sang tensorflow.lite (neu co).

Cai dependency toi thieu tren RPi (chon 1 trong 2, uu tien tflite-runtime vi
nhe hon nhieu so voi full tensorflow):
    pip install tflite-runtime
    # hoac (neu tflite-runtime khong co wheel cho kien truc/Python cua may):
    pip install ai-edge-litert

Usage tren RPi (sau khi da scp cac file .tflite vao ~/dev_ws):
    cd ~/dev_ws
    python3 rpi_benchmark_tflite.py \
        --model togronet_stage_float32.tflite --name "TogroNet-Stage" --precision FP32 \
        --model togronet_stage_float16.tflite --name "TogroNet-Stage" --precision FP16 \
        --model togronet_stage_int8.tflite    --name "TogroNet-Stage" --precision INT8 \
        --model togronet_fruit_float32.tflite --name "TogroNet-Fruit" --precision FP32 \
        --model togronet_fruit_float16.tflite --name "TogroNet-Fruit" --precision FP16 \
        --model togronet_fruit_int8.tflite    --name "TogroNet-Fruit" --precision INT8 \
        --runs 50 --csv rpi_benchmark_results.csv

Hoac don gian nhat, chi can trai het cac file .tflite trong thu muc hien tai:
    python3 rpi_benchmark_tflite.py --glob "*.tflite" --runs 50 --csv rpi_benchmark_results.csv
"""

import argparse
import csv
import glob
import os
import resource
import time

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
    BACKEND = "tflite_runtime"
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
        BACKEND = "ai_edge_litert"
    except ImportError:
        from tensorflow.lite import Interpreter
        BACKEND = "tensorflow.lite"


def peak_rss_mb() -> float:
    # ru_maxrss: Linux tra ve KB (khong phai byte nhu macOS)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_dummy_input(detail) -> np.ndarray:
    shape = detail["shape"]
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.random.randint(info.min, info.max + 1, size=shape).astype(dtype)
    return np.random.rand(*shape).astype(dtype)


def benchmark_one(path: str, runs: int, warmup: int) -> dict:
    size_mb = os.path.getsize(path) / (1024 * 1024)

    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    x = make_dummy_input(input_details[0])

    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])

    mem_before = peak_rss_mb()
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        latencies.append((time.perf_counter() - t0) * 1000.0)
    mem_after = peak_rss_mb()

    latencies = np.array(latencies)
    mean_ms = float(latencies.mean())
    return dict(
        size_mb=size_mb,
        mem_mb=mem_after,  # peak RSS cua ca process tinh den luc nay
        mem_delta_mb=mem_after - mem_before,
        latency_ms=mean_ms,
        latency_p50_ms=float(np.median(latencies)),
        fps=1000.0 / mean_ms if mean_ms > 0 else float("nan"),
        input_shape=str(input_details[0]["shape"].tolist()),
        input_dtype=str(input_details[0]["dtype"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="Duong dan file .tflite (co the lap lai)")
    ap.add_argument("--name", action="append", default=[], help="Ten model tuong ung voi moi --model (theo thu tu)")
    ap.add_argument("--precision", action="append", default=[], help="FP32/FP16/INT8 tuong ung moi --model")
    ap.add_argument("--glob", default=None, help="Thay vi liet ke --model, quet toan bo file khop pattern nay")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--csv", default="rpi_benchmark_results.csv")
    args = ap.parse_args()

    print(f"[backend tflite: {BACKEND}]")

    if args.glob:
        paths = sorted(glob.glob(args.glob))
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        precisions = ["?"] * len(paths)
    else:
        paths = args.model
        names = args.name if args.name else [os.path.basename(p) for p in paths]
        precisions = args.precision if args.precision else ["?"] * len(paths)

    if not paths:
        print("Khong co file .tflite nao de benchmark. Dung --model hoac --glob.")
        return

    rows = []
    for path, name, prec in zip(paths, names, precisions):
        print(f"Benchmark: {name} [{prec}] -> {path}")
        r = benchmark_one(path, args.runs, args.warmup)
        r.update(model=name, precision=prec, path=path)
        rows.append(r)
        print(
            f"  size={r['size_mb']:.2f}MB  mem_peak={r['mem_mb']:.1f}MB  "
            f"latency={r['latency_ms']:.2f}ms  fps={r['fps']:.2f}"
        )

    fieldnames = ["model", "precision", "path", "size_mb", "mem_mb", "mem_delta_mb",
                  "latency_ms", "latency_p50_ms", "fps", "input_shape", "input_dtype"]
    # Ghi noi (append), khong ghi de -- de co the goi script nhieu lan, moi lan
    # 1 process rieng (--model don le), va cong don ket qua vao cung 1 file CSV.
    # mem_mb (ru_maxrss) chi dung neu MOI PROCESS chi benchmark DUNG 1 model,
    # vi no la RAM dinh CUA CA PROCESS tu luc khoi dong, khong reset giua cac
    # model neu goi qua --glob/nhieu --model trong cung 1 lan chay.
    file_exists = os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)
    print(f"\nDa ghi noi vao: {args.csv}")


if __name__ == "__main__":
    main()
