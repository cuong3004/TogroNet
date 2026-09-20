from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


# =========================
# CONFIG
# =========================

OUTPUT_DIR = Path(r"C:\Users\HH\Downloads\tomato_ssl_dataset")
DATA_DIR = Path(r"C:\Users\HH\Downloads\Open_Images_V6")

IMAGE_SIZE = 416
NUM_WORKERS = 2

MAX_TRAIN_IMAGES = 300000
MAX_VAL_IMAGES = 5000

CLASSES = """
Tomato
Plant
Houseplant
Flower
Flowerpot
Tree
Vegetable
Fruit
Bell pepper
Cucumber
""".strip().splitlines()


# =========================
# PATHS
# =========================

CLASS_DESC_CSV = DATA_DIR / "oidv6-class-descriptions (1).csv"

TRAIN_BBOX_CSV = (
    DATA_DIR
    / "oidv6-train-annotations-bbox"
    / "oidv6-train-annotations-bbox.csv"
)

VAL_BBOX_CSV = DATA_DIR / "validation-annotations-bbox.csv"

TRAIN_IMAGE_URL_CSV = (
    DATA_DIR
    / "train-images-boxable-with-rotation"
    / "train-images-boxable-with-rotation.csv"
)

VAL_IMAGE_URL_CSV = DATA_DIR / "validation-images-with-rotation.csv"


# =========================
# UTILS
# =========================

def create_yolo_dirs(split: str):
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def load_class_mapping():
    class_df = pd.read_csv(
        CLASS_DESC_CSV,
        header=None,
        names=["LabelName", "ClassName"]
    )

    class_df = class_df[class_df["ClassName"].isin(CLASSES)].copy()
    class_df["order"] = class_df["ClassName"].apply(lambda name: CLASSES.index(name))
    class_df = class_df.sort_values("order")

    found_classes = class_df["ClassName"].tolist()
    missing_classes = sorted(set(CLASSES) - set(found_classes))

    print("Found classes:", len(found_classes))
    print("Missing classes:", missing_classes)

    class_to_id = {
        class_name: idx
        for idx, class_name in enumerate(found_classes)
    }

    label_to_id = {
        row.LabelName: class_to_id[row.ClassName]
        for row in class_df.itertuples()
    }

    print("Class map:", class_to_id)

    return label_to_id


def load_bbox_data(csv_path: Path, label_to_id: dict, max_images: int):
    bbox_df = pd.read_csv(csv_path)

    bbox_df = bbox_df[bbox_df["LabelName"].isin(label_to_id)]

    selected_image_ids = (
        bbox_df["ImageID"]
        .drop_duplicates()
        .head(max_images)
    )

    bbox_df = bbox_df[bbox_df["ImageID"].isin(selected_image_ids)]

    print(csv_path.name, "images:", bbox_df["ImageID"].nunique())
    print(csv_path.name, "boxes:", len(bbox_df))

    return bbox_df


def load_image_urls(csv_path: Path, bbox_df: pd.DataFrame):
    url_df = pd.read_csv(csv_path)

    image_ids = set(bbox_df["ImageID"].unique())

    url_df = url_df[url_df["ImageID"].isin(image_ids)]

    return url_df[["ImageID", "OriginalURL"]]


def build_yolo_label_map(bbox_df: pd.DataFrame, label_to_id: dict):
    label_map = {}

    for image_id, rows in bbox_df.groupby("ImageID"):
        lines = []

        for row in rows.itertuples():
            class_id = label_to_id[row.LabelName]

            x_center = (row.XMin + row.XMax) / 2
            y_center = (row.YMin + row.YMax) / 2
            width = row.XMax - row.XMin
            height = row.YMax - row.YMin

            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        label_map[image_id] = "\n".join(lines)

    return label_map


def download_and_save_image(session, image_url: str, image_path: Path):
    response = session.get(image_url, timeout=15)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image.save(image_path, quality=90)


def process_one_image(row, split: str, label_map: dict):
    image_id = row.ImageID
    image_url = row.OriginalURL

    image_path = OUTPUT_DIR / "images" / split / f"{image_id}.jpg"
    label_path = OUTPUT_DIR / "labels" / split / f"{image_id}.txt"

    try:
        with requests.Session() as session:
            if not image_path.exists():
                download_and_save_image(session, image_url, image_path)

        label_path.write_text(label_map[image_id], encoding="utf-8")

        return True, image_id, ""

    except Exception as error:
        return False, image_id, str(error)


def download_split(url_df: pd.DataFrame, bbox_df: pd.DataFrame, split: str, label_to_id: dict):
    create_yolo_dirs(split)

    label_map = build_yolo_label_map(bbox_df, label_to_id)
    rows = list(url_df.itertuples(index=False))

    success_count = 0
    failed_items = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_one_image, row, split, label_map)
            for row in rows
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=split):
            success, image_id, error = future.result()

            if success:
                success_count += 1
            else:
                failed_items.append((image_id, error))

    failed_log_path = OUTPUT_DIR / f"{split}_failed.txt"

    with failed_log_path.open("w", encoding="utf-8") as f:
        for image_id, error in failed_items:
            f.write(f"{image_id}\t{error}\n")

    print(f"{split}: success={success_count}, failed={len(failed_items)}")
    print("Failed log:", failed_log_path)


# =========================
# MAIN
# =========================

def main():
    label_to_id = load_class_mapping()

    train_bbox = load_bbox_data(TRAIN_BBOX_CSV, label_to_id, MAX_TRAIN_IMAGES)
    val_bbox = load_bbox_data(VAL_BBOX_CSV, label_to_id, MAX_VAL_IMAGES)

    train_urls = load_image_urls(TRAIN_IMAGE_URL_CSV, train_bbox)
    val_urls = load_image_urls(VAL_IMAGE_URL_CSV, val_bbox)

    download_split(train_urls, train_bbox, "train", label_to_id)
    download_split(val_urls, val_bbox, "val", label_to_id)


if __name__ == "__main__":
    main()