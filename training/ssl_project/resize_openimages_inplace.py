"""Resize toàn bộ ảnh open-images-v7 IN-PLACE (ghi đè đúng file gốc, không sinh
thêm file/thư mục output) sao cho cạnh dài nhất <= --max-side, giữ nguyên tỉ lệ
khung hình (chỉ scale đều, không crop/stretch). Ảnh đã có cạnh dài <= --max-side
sẽ được bỏ qua nguyên vẹn (không upscale).

Vì thao tác này ghi đè không thể hoàn tác trên ~104GB dữ liệu gốc, mỗi ảnh được
resize ra file tạm cùng thư mục rồi os.replace() đè lên file gốc theo kiểu
atomic -- nếu máy mất điện/bị kill giữa chừng thì tại thời điểm đó mỗi ảnh hoặc
còn nguyên bản gốc, hoặc đã thành bản đã resize hoàn chỉnh, không bao giờ có
file dở dang/hỏng. Vì điều kiện bỏ qua ảnh đã <= max-side, chạy lại script sau
khi bị ngắt giữa chừng sẽ tự resume đúng chỗ mà không cần theo dõi trạng thái
riêng.

Usage:
    # xem thử sẽ resize bao nhiêu ảnh, không ghi gì cả
    python3 resize_openimages_inplace.py --dry-run

    # chạy thật trên cả train + validation (mặc định)
    python3 resize_openimages_inplace.py

    # chỉ 1 thư mục, ít worker hơn
    python3 resize_openimages_inplace.py --dirs /home/agi/fiftyone/open-images-v7/validation/data --workers 4
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

DEFAULT_DIRS = [
    "/home/agi/fiftyone/open-images-v7/train/data",
    "/home/agi/fiftyone/open-images-v7/validation/data",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_image_paths(dirs: list[str]):
    for d in dirs:
        root = Path(d)
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p


def process_one(path_str: str, max_side: int, quality: int, dry_run: bool):
    path = Path(path_str)
    try:
        with Image.open(path) as im:
            w, h = im.size
            long_side = max(w, h)

            if long_side <= max_side:
                return ("skipped", path_str, w, h, w, h)

            scale = max_side / long_side
            new_w = max(1, round(w * scale))
            new_h = max(1, round(h * scale))

            if dry_run:
                return ("would_resize", path_str, w, h, new_w, new_h)

            im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            resized = im.resize((new_w, new_h), Image.LANCZOS)

            tmp_path = path.with_suffix(path.suffix + ".tmp")
            save_kwargs = {}
            fmt = (im.format or "JPEG")
            if fmt.upper() in ("JPEG", "JPG"):
                save_kwargs = dict(quality=quality, optimize=True)

            resized.save(tmp_path, format=fmt, **save_kwargs)
            os.replace(tmp_path, path)

            return ("resized", path_str, w, h, new_w, new_h)

    except Exception as e:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return ("error", path_str, str(e), None, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                     help="Các thư mục ảnh cần xử lý (mặc định: train + validation của open-images-v7)")
    ap.add_argument("--max-side", type=int, default=480,
                     help="Cạnh dài nhất sau resize (mặc định 480)")
    ap.add_argument("--quality", type=int, default=95,
                     help="JPEG quality khi lưu đè (mặc định 95)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                     help="Số tiến trình song song")
    ap.add_argument("--dry-run", action="store_true",
                     help="Chỉ đếm/báo cáo, không ghi đè file nào")
    ap.add_argument("--error-log", default="resize_openimages_errors.txt",
                     help="File ghi lại đường dẫn các ảnh xử lý lỗi")
    args = ap.parse_args()

    for d in args.dirs:
        if not Path(d).is_dir():
            print(f"Không tìm thấy thư mục: {d}", file=sys.stderr)
            sys.exit(1)

    print("Đang liệt kê danh sách ảnh...")
    paths = [str(p) for p in iter_image_paths(args.dirs)]
    print(f"Tổng số ảnh: {len(paths)}")

    counts = {"resized": 0, "would_resize": 0, "skipped": 0, "error": 0}
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, p, args.max_side, args.quality, args.dry_run): p
            for p in paths
        }
        with tqdm(total=len(futures), unit="img") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                status = result[0]
                counts[status] += 1
                if status == "error":
                    errors.append(f"{result[1]}\t{result[2]}")
                pbar.set_postfix(counts)
                pbar.update(1)

    print()
    print("=== KẾT QUẢ ===")
    for k, v in counts.items():
        print(f"{k}: {v}")

    if errors:
        Path(args.error_log).write_text("\n".join(errors) + "\n")
        print(f"\nĐã ghi {len(errors)} lỗi vào {args.error_log}")


if __name__ == "__main__":
    main()
