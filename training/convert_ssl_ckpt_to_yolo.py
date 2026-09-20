import argparse
import re
import torch
from ultralytics import YOLO

# Backbone la cac layer index 0-10 trong section "backbone:" cua togroth_*.yaml
# (Conv,Conv,C3k2,Conv,C3k2,Conv,C3k2,Conv,C3k2,SPPF,C2PSA) -- tu layer 11 tro
# di la neck (FPN/PAN) roi Detect head (layer 23). Dung khi muon chi lay
# backbone tu SSL, giu nguyen neck+head tu checkpoint COCO (--base_yolo), vi
# loss SSL hien tai (Barlow Twins tren P5 da AdaptiveAvgPool2d(1)) chi toi uu
# bat bien muc toan anh, khong giu thong tin khong gian ma neck detect can.
BACKBONE_MAX_LAYER = 10
LAYER_IDX_RE = re.compile(r"^model\.(\d+)\.")


def load_lightning_state_dict(ckpt_path):
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" not in ckpt:
        raise RuntimeError("Không thấy key 'state_dict' trong Lightning checkpoint")

    return ckpt["state_dict"]


def convert_ssl_to_yolo(
    ssl_ckpt_path,
    base_yolo_weight,
    output_weight,
    only_backbone=False,
):
    print("Loading base YOLO:", base_yolo_weight)
    yolo = YOLO(base_yolo_weight)

    yolo_model = yolo.model
    yolo_sd = yolo_model.state_dict()

    print("Loading SSL checkpoint:", ssl_ckpt_path)
    ssl_sd = load_lightning_state_dict(ssl_ckpt_path)

    copied = []
    skipped = []

    new_sd = yolo_sd.copy()

    for ssl_key, ssl_tensor in ssl_sd.items():

        # Lightning key:
        # encoder.model.model.0.conv.weight
        #
        # YOLO key:
        # model.0.conv.weight

        if not ssl_key.startswith("encoder.model."):
            skipped.append((ssl_key, "not encoder"))
            continue

        yolo_key = ssl_key.replace("encoder.model.", "", 1)

        if yolo_key not in yolo_sd:
            skipped.append((ssl_key, "key not found"))
            continue

        if ssl_tensor.shape != yolo_sd[yolo_key].shape:
            skipped.append(
                (
                    ssl_key,
                    f"shape mismatch {tuple(ssl_tensor.shape)} vs {tuple(yolo_sd[yolo_key].shape)}",
                )
            )
            continue

        if only_backbone:
            m = LAYER_IDX_RE.match(yolo_key)
            if not m or int(m.group(1)) > BACKBONE_MAX_LAYER:
                skipped.append((ssl_key, "only-backbone: layer thuoc neck/head"))
                continue

        new_sd[yolo_key] = ssl_tensor
        copied.append((ssl_key, yolo_key))

    shape_mismatches = [s for s in skipped if s[1].startswith("shape mismatch")]

    print("=" * 80)
    print("COPY SUMMARY")
    print("=" * 80)
    print("Copied:", len(copied))
    print("Skipped:", len(skipped), f"(trong đó {len(shape_mismatches)} do lệch shape)")

    if len(shape_mismatches) > 0.1 * len(ssl_sd):
        print(
            f"\n[CẢNH BÁO] {len(shape_mismatches)}/{len(ssl_sd)} tensor bị bỏ qua do LỆCH SHAPE "
            f"(>10% checkpoint) -- rất có thể --base_yolo không đúng kiến trúc/width mà SSL "
            f"checkpoint này được train. Chỉ {len(copied)} tensor thực sự được chuyển giao, "
            f"phần còn lại của model vẫn giữ nguyên trọng số gốc từ --base_yolo. Kiểm tra lại "
            f"đúng file cfg/width trước khi dùng output này."
        )

    print("\nCopied keys:")
    for a, b in copied[:50]:
        print(f"{a}  -->  {b}")

    if len(copied) > 50:
        print(f"... {len(copied) - 50} more")

    print("\nSkipped examples:")
    for k, reason in skipped[:30]:
        print(f"{k}  |  {reason}")

    yolo_model.load_state_dict(new_sd, strict=False)

    print("\nSaving new YOLO weight:", output_weight)
    yolo.save(output_weight)

    print("Done!")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ssl_ckpt",
        type=str,
        required=True,
        help="Path to Lightning last.ckpt",
    )

    parser.add_argument(
        "--base_yolo",
        type=str,
        default="yolo26n.pt",
        help="Original YOLO weight",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="yolo26n_ssl_init.pt",
        help="Output YOLO weight",
    )

    parser.add_argument(
        "--only-backbone",
        action="store_true",
        help=(
            "Chi chuyen giao trong so SSL cho backbone (layer 0-10), giu nguyen "
            "neck+head tu --base_yolo. Dung khi SSL (loss tren P5 da pool toan "
            "cuc) lam giam mAP do pha hong dac trung khong gian cua neck."
        ),
    )

    args = parser.parse_args()

    convert_ssl_to_yolo(
        ssl_ckpt_path=args.ssl_ckpt,
        base_yolo_weight=args.base_yolo,
        output_weight=args.output,
        only_backbone=args.only_backbone,
    )


if __name__ == "__main__":
    main()



# python3 convert_ssl_ckpt_to_yolo.py \
#     --ssl_ckpt /home/agi/thesis_code/ssl_project/lightning_logs/version_1/checkpoints/last.ckpt \
#     --base_yolo yolo26n.pt \
#     --output yolo26n_ssl_init.pt