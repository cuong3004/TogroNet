#!/usr/bin/env python3
"""Pha 2 - xu ly toan bo anh cua MOT luot sau khi robot da ve.

Thu tu moi anh: tien xu ly -> chay Stage -> neu trang thai thuoc nhom co
qua (hinh_thanh_qua / qua_chuyen_chin) thi chay Fruit -> dem so hop con lai
theo tung lop. Do la toan bo phep dem. Khong cong don qua nhieu anh, khong
tong hop theo cay/vi tri, khong tracking.

Thu tu lop lay tu metadata nhung trong file .tflite (Ultralytics export),
khong hardcode.
"""

import argparse
import csv
import json
import math
import os
import zipfile
from collections import defaultdict

import cv2
import yaml
import numpy as np

STAGE_VN_KEY = {
    'vegetative_growth': 'sinh_truong_sinh_duong',
    'flowering': 'ra_hoa',
    'fruit_development': 'hinh_thanh_qua',
    'fruit_ripening': 'qua_chuyen_chin',
}
FRUIT_TRIGGER_STAGES = {'hinh_thanh_qua', 'qua_chuyen_chin'}

# Anh chup trong luc di chuyen (khong co cay trong khung) van co the roi
# vao ban kinh 0.8m cua mot waypoint (loc theo khong gian), nhung Stage se
# cho confidence thap vi khung hinh ngoai phan phoi huan luyen. Duoi nguong
# nay -> coi la khong ro/khong co cay, khong tinh vao thong ke.
STAGE_UNCERTAIN = 'khong_ro'

# Anh bi bo qua vi da du dai dien cho vi tri do (chong trung lap).
STAGE_SKIPPED_DUPLICATE = 'bo_qua_trung_lap'

FRUIT_VN_KEY = {
    'green': 'qua_xanh',
    'half_ripened': 'qua_chuyen_mau',
    'fully_ripened': 'qua_chin',
}

END2END_COLUMNS = 6  # x1, y1, x2, y2, conf, cls


def load_ultralytics_metadata(model_path):
    """Doc metadata.json nhung trong file .tflite (Ultralytics export)."""
    try:
        with zipfile.ZipFile(model_path) as z:
            data = json.loads(z.read('metadata.json').decode('utf-8'))
    except (zipfile.BadZipFile, KeyError) as exc:
        raise RuntimeError(
            f"Khong doc duoc metadata nhung trong '{model_path}': {exc}. "
            'Khong the xac dinh thu tu lop mot cach an toan, dung lai '
            'thay vi doan.'
        ) from exc
    names = {int(k): v for k, v in data['names'].items()}
    return data, names


def letterbox(image, new_h, new_w, color=(114, 114, 114)):
    h, w = image.shape[:2]
    r = min(new_h / h, new_w / w)
    rw, rh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_w - rw, new_h - rh
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def _load_interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError:
        pass
    try:
        # tflite-runtime khong co wheel cho Python 3.13/aarch64. ai-edge-litert
        # la ban ke thua chinh thuc, cung API Interpreter/load_delegate.
        from ai_edge_litert.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError:
        pass
    try:
        from tensorflow.lite.python.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError as exc:
        raise ImportError(
            "Can mot trong: 'pip install ai-edge-litert' (khuyen nghi tren "
            "Python 3.13/aarch64), 'python3-tflite-runtime' (Coral apt repo), "
            f"hoac tensorflow day du. Chi tiet: {exc}"
        ) from exc


class TFLiteModel:

    def __init__(self, model_path, use_edgetpu=False, label='model'):
        self.label = label
        self.metadata, self.names = load_ultralytics_metadata(model_path)
        self.task = self.metadata.get('task')

        Interpreter, load_delegate = _load_interpreter_class()
        self.interpreter = None
        if use_edgetpu:
            try:
                delegate = load_delegate('libedgetpu.so.1')
                self.interpreter = Interpreter(
                    model_path=model_path, experimental_delegates=[delegate]
                )
                print(f'[{label}] Dung Edge TPU delegate.')
            except Exception as exc:
                print(f'[{label}] Khong nap duoc Edge TPU delegate ({exc}), chay CPU.')
        if self.interpreter is None:
            self.interpreter = Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        in_shape = self.input_details[0]['shape']
        self.input_h, self.input_w = int(in_shape[1]), int(in_shape[2])

    def _quantize_input(self, rgb_uint8):
        detail = self.input_details[0]
        dtype = detail['dtype']
        scale, zero_point = detail['quantization']
        float_img = rgb_uint8.astype(np.float32) / 255.0
        if dtype in (np.uint8, np.int8) and scale not in (0, 0.0):
            q = np.round(float_img / scale + zero_point)
            info = np.iinfo(dtype)
            return np.clip(q, info.min, info.max).astype(dtype)
        # Vo boc float32 (quan sat thuc te tren 2 model nay: input/output la
        # float32 du ten file co '_int8' - luong tu hoa nam ben trong graph).
        return float_img.astype(dtype)

    def _dequantize_output(self, index):
        detail = self.output_details[index]
        raw = self.interpreter.get_tensor(detail['index'])
        scale, zero_point = detail['quantization']
        if scale not in (0, 0.0):
            return (raw.astype(np.float32) - zero_point) * scale
        return raw.astype(np.float32)

    def infer(self, frame_bgr):
        padded = letterbox(frame_bgr, self.input_h, self.input_w)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = self._quantize_input(rgb)[None, ...]
        self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
        self.interpreter.invoke()
        return [self._dequantize_output(i) for i in range(len(self.output_details))]


def run_stage(model, frame_bgr):
    outputs = model.infer(frame_bgr)
    probs = outputs[0].reshape(-1)
    idx = int(np.argmax(probs))
    name_en = model.names[idx]
    if name_en not in STAGE_VN_KEY:
        raise RuntimeError(
            f"Ten lop '{name_en}' tu metadata Stage khong nam trong bang anh xa "
            f'da xac nhan truoc: {sorted(STAGE_VN_KEY)}'
        )
    return STAGE_VN_KEY[name_en], name_en, float(probs[idx])


def _pick_2d_tensor(outputs):
    for out in outputs:
        arr = np.squeeze(out, axis=0) if out.shape[0] == 1 else out
        if arr.ndim == 2:
            return arr
    return None


def _decode_end2end(arr, conf_threshold):
    kept = []
    for row in arr:
        x1, y1, x2, y2, conf, cls = row[:6]
        if conf < conf_threshold:
            continue
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0 and conf == 0:
            continue
        kept.append((int(round(float(cls))), float(conf)))
    return kept


def _decode_raw_grid(arr, num_classes, conf_threshold, iou_threshold):
    boxes = arr[:, :4]
    scores_all = arr[:, 4:4 + num_classes]
    class_ids = np.argmax(scores_all, axis=1)
    class_scores = scores_all[np.arange(len(scores_all)), class_ids]

    mask = class_scores >= conf_threshold
    boxes, class_ids, class_scores = boxes[mask], class_ids[mask], class_scores[mask]
    if len(boxes) == 0:
        return []

    xywh = np.empty_like(boxes)
    xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xywh[:, 2] = boxes[:, 2]
    xywh[:, 3] = boxes[:, 3]

    indices = cv2.dnn.NMSBoxes(
        xywh.tolist(), class_scores.tolist(), conf_threshold, iou_threshold
    )
    indices = np.array(indices).reshape(-1) if len(indices) else np.array([], dtype=int)
    return [(int(class_ids[i]), float(class_scores[i])) for i in indices]


def decode_fruit_output(outputs, num_classes, conf_threshold, iou_threshold):
    arr = _pick_2d_tensor(outputs)
    if arr is None:
        raise RuntimeError('Khong tim thay tensor dau ra 2 chieu cho Fruit model.')

    raw_dim = 4 + num_classes
    if END2END_COLUMNS in arr.shape:
        if arr.shape[1] != END2END_COLUMNS:
            arr = arr.T
        return _decode_end2end(arr, conf_threshold)

    if raw_dim in arr.shape:
        if arr.shape[1] != raw_dim:
            arr = arr.T
        return _decode_raw_grid(arr, num_classes, conf_threshold, iou_threshold)

    raise RuntimeError(
        f'Khong nhan dang duoc dinh dang dau ra Fruit: shape={arr.shape}, '
        f'ky vong mot chieu = {END2END_COLUMNS} (end2end) hoac {raw_dim} (raw grid). '
        "Chay lai voi '--dump-io' de kiem tra tensor that su."
    )


def run_fruit(model, frame_bgr, conf_threshold, iou_threshold):
    for name_en in model.names.values():
        if name_en not in FRUIT_VN_KEY:
            raise RuntimeError(
                f"Ten lop '{name_en}' tu metadata Fruit khong nam trong bang anh xa "
                f'da xac nhan truoc: {sorted(FRUIT_VN_KEY)}'
            )

    outputs = model.infer(frame_bgr)
    detections = decode_fruit_output(
        outputs, len(model.names), conf_threshold, iou_threshold
    )

    counts = {key: 0 for key in FRUIT_VN_KEY.values()}
    for cls_id, _score in detections:
        name_en = model.names.get(cls_id)
        if name_en is not None:
            counts[FRUIT_VN_KEY[name_en]] += 1
    return counts, sum(counts.values())


def load_target_yaws(waypoints_file):
    """Doc yaw muc tieu cua tung vi tri quan sat tu waypoints.yaml.

    Tra ve dict {position_id: yaw_rad}. Neu khong doc duoc file, tra ve {}
    (goi noi se bo qua buoc loc theo yaw, giu hanh vi cu).
    """
    try:
        with open(waypoints_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return {
            wp['id']: float(wp['yaw'])
            for wp in data.get('observation_waypoints', [])
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f'[canh bao] khong doc duoc waypoints_file de loc theo yaw: {exc}')
        return {}


def _yaw_diff_deg(a_rad, b_rad):
    """Chenh lech goc nho nhat giua 2 yaw (radian), tra ve do, luon >= 0."""
    diff = math.degrees(a_rad - b_rad)
    return abs((diff + 180.0) % 360.0 - 180.0)


def select_representative_indices(
    records, max_per_position, target_yaws=None, yaw_tolerance_deg=25.0,
):
    """
    Voi moi vi tri that (khong tinh qua_duong):
    1. Neu co yaw muc tieu cho vi tri do (tu waypoints.yaml), chi xet cac
       anh co yaw luc chup lech khong qua yaw_tolerance_deg so voi yaw muc
       tieu - loai cac anh chup luc con dang tien vao/di ngang qua (huong
       may quay chua dung vao cay). Neu loc xong khong con anh nao (vd
       robot chua bao gio dung dung huong), fallback dung lai TOAN BO anh
       cua vi tri do de tranh mat trang du lieu.
    2. Neu so anh (sau loc) van vuot max_per_position: GIAI DEU theo thu
       tu trong manifest (khong uu tien "gan dung huong nhat theo so" -
       da kiem chung TREN ROBOT THAT: anh gan dung yaw muc tieu nhat co
       the chi la vai khung hinh lien tiep cung 1 khoanh khac, con anh o
       ria nguong lai chup dung luc qua lo ro nhat. Giai deu giu da dang
       goc nhin/thoi diem hon la dua theo 1 con so yaw don le).

    Tra ve set() cac index (trong `records`) duoc GIU LAI de chay day du.
    Anh vi tri qua_duong luon duoc giu (khong bi gioi han o day, vi ban
    than qua_duong da bi loai khoi results_table.csv roi).
    """
    target_yaws = target_yaws or {}
    by_position = defaultdict(list)
    for i, rec in enumerate(records):
        if rec['position'] != 'qua_duong':
            by_position[rec['position']].append(i)

    keep = set(i for i, rec in enumerate(records) if rec['position'] == 'qua_duong')

    for position, indices in by_position.items():
        target_yaw = target_yaws.get(position)
        if target_yaw is not None:
            yaw_matched = [
                i for i in indices
                if _yaw_diff_deg(records[i]['pose']['yaw'], target_yaw) <= yaw_tolerance_deg
            ]
            indices = yaw_matched if yaw_matched else indices

        n = len(indices)
        if n <= max_per_position or max_per_position <= 0:
            keep.update(indices)
            continue
        if max_per_position == 1:
            chosen = [indices[0]]
        else:
            # Giai deu theo thu tu manifest (giu da dang thoi diem/goc nhin).
            chosen = [
                indices[round(k * (n - 1) / (max_per_position - 1))]
                for k in range(max_per_position)
            ]
        keep.update(chosen)

    return keep


def dump_io(model_path, label, use_edgetpu):
    model = TFLiteModel(model_path, use_edgetpu=use_edgetpu, label=label)
    print(f'== {label} ({model_path}) ==')
    print('task:', model.task, ' names:', model.names)
    print('input_details:')
    for d in model.input_details:
        print(' ', d)
    print('output_details:')
    for d in model.output_details:
        print(' ', d)


def main():
    parser = argparse.ArgumentParser(
        description='Pha 2: TogroNet-Stage + dieu kien TogroNet-Fruit tren anh mot luot.'
    )
    parser.add_argument('run_dir', help='Thu muc luot (chua images/ va manifest.jsonl)')
    parser.add_argument(
        '--stage-model', default=os.path.expanduser('~/dev_ws/stage_last_int8.tflite')
    )
    parser.add_argument(
        '--fruit-model', default=os.path.expanduser('~/dev_ws/fruit_last_int8.tflite')
    )
    parser.add_argument('--edgetpu', action='store_true', help='Dung Edge TPU delegate neu co')
    parser.add_argument('--conf-fruit', type=float, default=0.25)
    parser.add_argument('--iou-fruit', type=float, default=0.45)
    parser.add_argument(
        '--stage-confidence-threshold', type=float, default=0.7,
        help=(
            'Duoi nguong nay, Stage bi coi la khong ro/khong co cay '
            '(anh chup luc di chuyen roi vao ban kinh waypoint nhung khong '
            'thay cay) - khong tinh vao thong ke, khong chay Fruit.'
        ),
    )
    parser.add_argument(
        '--max-per-position', type=int, default=5,
        help=(
            'So anh toi da xu ly cho moi vi tri that (uu tien anh dung '
            'huong nhat neu co yaw muc tieu, khong tinh qua_duong). '
            '0 = khong gioi han.'
        ),
    )
    parser.add_argument(
        '--waypoints-file',
        default=os.path.expanduser(
            '~/dev_ws/install/tomato_survey/share/tomato_survey/config/waypoints.yaml'
        ),
        help=(
            'File waypoints.yaml de lay yaw muc tieu cua tung vi tri, dung '
            'loc bot anh chup luc con dang tien vao/di ngang qua (chua quay '
            'dung huong). Neu khong ton tai, bo qua buoc loc theo yaw.'
        ),
    )
    parser.add_argument(
        '--yaw-tolerance-deg', type=float, default=25.0,
        help=(
            'Chi giu anh co yaw luc chup lech khong qua nguong nay so voi '
            'yaw muc tieu cua vi tri (waypoints.yaml). Neu khong anh nao '
            'dat, fallback dung lai toan bo anh cua vi tri do.'
        ),
    )
    parser.add_argument(
        '--require-complete', action='store_true',
        help='Dung neu run_summary.json bao run_complete=false',
    )
    parser.add_argument(
        '--dump-io', action='store_true',
        help='Chi in input/output tensor cua 2 model roi thoat (kiem tra truoc khi tin ket qua)',
    )
    args = parser.parse_args()

    if args.dump_io:
        dump_io(args.stage_model, 'Stage', args.edgetpu)
        dump_io(args.fruit_model, 'Fruit', args.edgetpu)
        return

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    manifest_path = os.path.join(run_dir, 'manifest.jsonl')
    images_dir = os.path.join(run_dir, 'images')
    summary_path = os.path.join(run_dir, 'run_summary.json')

    if args.require_complete and os.path.isfile(summary_path):
        with open(summary_path, encoding='utf-8') as f:
            summary = json.load(f)
        if not summary.get('run_complete'):
            raise SystemExit(
                'run_summary.json bao run_complete=false, dung lai theo --require-complete.'
            )

    records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    target_yaws = load_target_yaws(args.waypoints_file)
    keep_indices = select_representative_indices(
        records, args.max_per_position,
        target_yaws=target_yaws, yaw_tolerance_deg=args.yaw_tolerance_deg,
    )
    skipped_duplicate_count = len(records) - len(keep_indices)

    stage_model = TFLiteModel(args.stage_model, use_edgetpu=args.edgetpu, label='Stage')
    fruit_model = None

    inference_path = os.path.join(run_dir, 'inference.jsonl')
    table_path = os.path.join(run_dir, 'results_table.csv')

    stage_counts_summary = defaultdict(int)
    fruit_triggered = 0
    uncertain_count = 0

    with open(inference_path, 'w', encoding='utf-8') as inf_f, \
            open(table_path, 'w', newline='', encoding='utf-8') as csv_f:

        writer = csv.writer(csv_f)
        writer.writerow([
            'image_id', 'position', 'target_waypoint_index', 'captured_at',
            'stage', 'stage_confidence',
            'qua_xanh', 'qua_chuyen_mau', 'qua_chin', 'fruit_total',
        ])

        for idx, rec in enumerate(records):
            if idx not in keep_indices:
                # Da du anh dai dien cho vi tri nay, bo qua khong chay model
                # (tranh trung lap qua nhieu anh gan giong nhau).
                inf_f.write(json.dumps({
                    'image_id': rec['image_id'],
                    'position': rec['position'],
                    'target_waypoint_index': rec['target_waypoint_index'],
                    'captured_at': rec['captured_at'],
                    'stage': STAGE_SKIPPED_DUPLICATE,
                    'stage_name_en': None,
                    'stage_confidence': None,
                    'fruit_counts': 'NA',
                    'fruit_total': 'NA',
                }, ensure_ascii=False) + '\n')
                continue

            image_path = os.path.join(images_dir, rec['image_id'])
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"[canh bao] khong doc duoc anh {rec['image_id']}, bo qua.")
                continue

            stage_key, stage_name_en, stage_conf = run_stage(stage_model, frame)

            if stage_conf < args.stage_confidence_threshold:
                # Anh chup luc di chuyen, roi vao ban kinh waypoint nhung
                # khong thay cay ro rang trong khung -> khong tinh vao
                # thong ke, khong chay Fruit du du doan tho la gi.
                uncertain_count += 1
                stage_key = STAGE_UNCERTAIN
                fruit_counts, fruit_total = 'NA', 'NA'
            else:
                stage_counts_summary[stage_key] += 1
                if stage_key in FRUIT_TRIGGER_STAGES:
                    if fruit_model is None:
                        fruit_model = TFLiteModel(
                            args.fruit_model, use_edgetpu=args.edgetpu, label='Fruit'
                        )
                    fruit_counts, fruit_total = run_fruit(
                        fruit_model, frame, args.conf_fruit, args.iou_fruit
                    )
                    fruit_triggered += 1
                else:
                    fruit_counts, fruit_total = 'NA', 'NA'

            inf_f.write(json.dumps({
                'image_id': rec['image_id'],
                'position': rec['position'],
                'target_waypoint_index': rec['target_waypoint_index'],
                'captured_at': rec['captured_at'],
                'stage': stage_key,
                'stage_name_en': stage_name_en,
                'stage_confidence': stage_conf,
                'fruit_counts': fruit_counts,
                'fruit_total': fruit_total,
            }, ensure_ascii=False) + '\n')

            if rec['position'] != 'qua_duong' and stage_key != STAGE_UNCERTAIN:
                writer.writerow([
                    rec['image_id'], rec['position'], rec['target_waypoint_index'],
                    rec['captured_at'], stage_key, f'{stage_conf:.4f}',
                    fruit_counts['qua_xanh'] if fruit_counts != 'NA' else 'NA',
                    fruit_counts['qua_chuyen_mau'] if fruit_counts != 'NA' else 'NA',
                    fruit_counts['qua_chin'] if fruit_counts != 'NA' else 'NA',
                    fruit_total,
                ])

    print(
        f'Da xu ly {len(records)} anh: '
        f'{len(keep_indices)} chay model, {skipped_duplicate_count} bo qua (trung lap vi tri), '
        f'{uncertain_count} khong ro/khong co cay (confidence < {args.stage_confidence_threshold}). '
        f'Fruit da chay tren {fruit_triggered} anh.'
    )
    for key, count in stage_counts_summary.items():
        print(f'  {key}: {count} anh')
    print(f'-> {inference_path}')
    print(f'-> {table_path}')


if __name__ == '__main__':
    main()
