"""Export 1 model sang 1 dinh dang tflite (float32/float16/int8), chay nhu 1
process rieng biet -- de RAM duoc giai phong hoan toan giua cac lan export,
tranh OOM khi don nhieu lan export chong len nhau trong cung 1 process."""

import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--precision", required=True, choices=["float32", "float16", "int8"])
parser.add_argument("--data", default=None)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--fraction", type=float, default=1.0,
                     help="Ti le tap calibration dung cho int8 (0-1), giup gioi han RAM")
args = parser.parse_args()

model = YOLO(args.weights)

kwargs = dict(format="tflite", imgsz=args.imgsz)
if args.precision == "float16":
    kwargs["half"] = True
elif args.precision == "int8":
    kwargs["int8"] = True
    kwargs["data"] = args.data
    kwargs["fraction"] = args.fraction

model.export(**kwargs)
print(f"DONE: {args.weights} [{args.precision}]")
