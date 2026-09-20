# convert_xml_to_yolo.py

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# =========================
# CONFIG
# =========================

SRC_IMAGES = Path("/home/agi/thesis_code/tomato_dataset/images")
SRC_XML = Path("/home/agi/thesis_code/tomato_dataset/annotations")

OUT_DIR = Path("/home/agi/thesis_code/tomato_yolo_dataset")

CLASS_MAP = {
    "tomato": 0
}

TRAIN_RATIO = 0.8
SEED = 42

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]


# =========================
# FUNCTIONS
# =========================

def voc_to_yolo(size, box):
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    return x_center, y_center, w, h


def convert_xml(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()

        if class_name not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[class_name]

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x, y, w, h = voc_to_yolo((img_w, img_h), (xmin, ymin, xmax, ymax))

        lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    txt_path.write_text("\n".join(lines))


def find_image(image_id):
    for ext in IMAGE_EXTS:
        img_path = SRC_IMAGES / f"{image_id}{ext}"
        if img_path.exists():
            return img_path
    return None


def prepare_dirs():
    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


# =========================
# MAIN
# =========================

prepare_dirs()

xml_files = list(SRC_XML.glob("*.xml"))
image_ids = [xml.stem for xml in xml_files]

random.seed(SEED)
random.shuffle(image_ids)

n_train = int(len(image_ids) * TRAIN_RATIO)
train_ids = image_ids[:n_train]
val_ids = image_ids[n_train:]

splits = {
    "train": train_ids,
    "val": val_ids
}

for split, ids in splits.items():
    for image_id in ids:
        img_path = find_image(image_id)
        xml_path = SRC_XML / f"{image_id}.xml"

        if img_path is None:
            print(f"Missing image: {image_id}")
            continue

        dst_img = OUT_DIR / "images" / split / img_path.name
        dst_txt = OUT_DIR / "labels" / split / f"{image_id}.txt"

        shutil.copy2(img_path, dst_img)
        convert_xml(xml_path, dst_txt)

# tạo data.yaml
yaml_text = f"""train: {OUT_DIR}/images/train
val: {OUT_DIR}/images/val

nc: 1
names: ['tomato']
"""

(OUT_DIR / "data.yaml").write_text(yaml_text)

print("Done!")
print(f"Dataset saved to: {OUT_DIR}")
print(f"Train images: {len(train_ids)}")
print(f"Val images: {len(val_ids)}")
print(f"YAML: {OUT_DIR / 'data.yaml'}")