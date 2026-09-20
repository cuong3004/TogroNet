#!/usr/bin/env python3
"""
Hệ thống giám sát giai đoạn sinh trưởng cây trồng.

- Camera trực tiếp: Raspberry Pi Camera V2 qua Picamera2/libcamera.
- MANUAL: dự đoán ngay trên ảnh mới nhất.
- AUTO: bật/tắt dự đoán định kỳ.
- Không lưu ảnh xuống ổ đĩa.
- Hỗ trợ checkpoint Ultralytics dạng classification hoặc detection.

Nhãn giai đoạn:
    1: Cây con
    2: Trưởng thành
    3: Ra hoa
    4: Có quả
    5: Quả chín
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from picamera2 import Picamera2
# from ultralytics import YOLO


# ============================================================
# CẤU HÌNH CỐ ĐỊNH
# ============================================================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 20.0
CAMERA_WARMUP_SEC = 1.0

GUI_WIDTH = 684
GUI_HEIGHT = 600
DISPLAY_WIDTH = 500
DISPLAY_HEIGHT = 360

# Đường dẫn checkpoint đã huấn luyện 5 giai đoạn.
MODEL_PATH = Path.home() / "models" / "growth_stage.pt"
MODEL_IMGSZ = 224
MODEL_DEVICE = "cpu"
MIN_CONFIDENCE = 0.50

# AUTO dự đoán sau mỗi khoảng thời gian này.
AUTO_PREDICT_INTERVAL_SEC = 2.0

# ROI trung tâm đưa vào mô hình.
ROI_WIDTH_RATIO = 0.60
ROI_HEIGHT_RATIO = 0.75
ROI_CENTER_X_RATIO = 0.50
ROI_CENTER_Y_RATIO = 0.50

# Kiểm tra nhanh xem ROI có đủ màu xanh hay không trước khi dự đoán.
# Có thể đặt USE_HSV_PLANT_GATE = False nếu mô hình tự xử lý nền tốt.
USE_HSV_PLANT_GATE = True
HSV_LOWER = np.array([25, 40, 30], dtype=np.uint8)
HSV_UPPER = np.array([100, 255, 255], dtype=np.uint8)
GREEN_RATIO_THRESHOLD = 0.06

# Mapping chuẩn nếu class model là 0..4 hoặc tên class là 1..5.
STAGE_LABELS = {
    0: "Cây con",
    1: "Trưởng thành",
    2: "Ra hoa",
    3: "Có quả",
    4: "Quả chín",
}

NUMERIC_NAME_TO_LABEL = {
    "1": "Cây con",
    "2": "Trưởng thành",
    "3": "Ra hoa",
    "4": "Có quả",
    "5": "Quả chín",
}


class GrowthMonitor:
    """Camera, tiền xử lý và suy luận mô hình trong các luồng riêng."""

    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.picam2: Optional[Picamera2] = None
        self.camera_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None

        self.latest_bgr: Optional[np.ndarray] = None
        self.latest_display_bgr: Optional[np.ndarray] = None

        self.model: Optional[YOLO] = None
        self.model_status = "Đang nạp mô hình"
        self.camera_status = "Đang khởi tạo camera"

        self.auto_enabled = False
        self.last_auto_request_time = 0.0
        self.inference_busy = False

        # Queue chỉ giữ yêu cầu mới nhất, tránh AUTO tạo backlog.
        self.inference_queue: queue.Queue[tuple[str, np.ndarray]] = queue.Queue(maxsize=1)

        self.predicted_stage = "Chưa dự đoán"
        self.predicted_class = "-"
        self.confidence = 0.0
        self.inference_ms = 0.0
        self.green_ratio = 0.0
        self.last_trigger = "-"
        self.prediction_count = 0
        self.last_error = "-"

        self.load_model()
        self.start_camera()
        self.start_inference_worker()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    def load_model(self) -> None:
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

            self.model = YOLO(str(MODEL_PATH))
            self.model_status = "Mô hình đã sẵn sàng"
        except Exception as exc:
            self.model = None
            self.model_status = "Lỗi mô hình"
            self.last_error = str(exc)

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------
    def start_camera(self) -> None:
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                    "format": "BGR888",
                },
                controls={"FrameRate": CAMERA_FPS},
                buffer_count=4,
            )
            self.picam2.configure(config)
            self.picam2.start()

            self.camera_thread = threading.Thread(
                target=self.camera_loop,
                name="picamera2_capture",
                daemon=True,
            )
            self.camera_thread.start()
        except Exception as exc:
            self.camera_status = "Lỗi camera"
            self.last_error = str(exc)

    def camera_loop(self) -> None:
        time.sleep(CAMERA_WARMUP_SEC)
        frame_period = 1.0 / max(CAMERA_FPS, 1.0)
        self.camera_status = "Camera hoạt động"

        while not self.stop_event.is_set():
            started = time.monotonic()

            try:
                if self.picam2 is None:
                    break

                # Picamera2 trả RGB888.
                rgb_frame = self.picam2.capture_array("main")
                if rgb_frame is None or rgb_frame.size == 0:
                    continue

                # OpenCV xử lý theo BGR.
                frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                self.process_frame(frame_bgr)

            except Exception as exc:
                self.camera_status = "Lỗi đọc camera"
                self.last_error = str(exc)
                time.sleep(0.3)

            remaining = frame_period - (time.monotonic() - started)
            if remaining > 0:
                self.stop_event.wait(remaining)

    def process_frame(self, frame_bgr: np.ndarray) -> None:
        x1, y1, x2, y2 = self.compute_roi(frame_bgr.shape[1], frame_bgr.shape[0])
        roi_bgr = frame_bgr[y1:y2, x1:x2]
        green_ratio = self.compute_green_ratio(roi_bgr)

        display = frame_bgr.copy()
        roi_color = (0, 170, 0) if green_ratio >= GREEN_RATIO_THRESHOLD else (0, 180, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), roi_color, 2)

        with self.result_lock:
            stage = self.predicted_stage
            confidence = self.confidence
            auto_enabled = self.auto_enabled
            busy = self.inference_busy

        cv2.putText(
            display,
            f"AUTO: {'ON' if auto_enabled else 'OFF'}",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "DANG DU DOAN" if busy else f"GIAI DOAN: {stage}",
            (14, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if confidence > 0.0:
            cv2.putText(
                display,
                f"TIN CAY: {confidence * 100:.1f}%",
                (14, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        with self.frame_lock:
            self.latest_bgr = frame_bgr.copy()
            self.latest_display_bgr = display
            self.green_ratio = green_ratio

        self.maybe_request_auto_prediction(roi_bgr)

    @staticmethod
    def compute_roi(width: int, height: int) -> tuple[int, int, int, int]:
        roi_w = int(width * ROI_WIDTH_RATIO)
        roi_h = int(height * ROI_HEIGHT_RATIO)
        cx = int(width * ROI_CENTER_X_RATIO)
        cy = int(height * ROI_CENTER_Y_RATIO)

        x1 = max(0, cx - roi_w // 2)
        y1 = max(0, cy - roi_h // 2)
        x2 = min(width, x1 + roi_w)
        y2 = min(height, y1 + roi_h)
        return x1, y1, x2, y2

    @staticmethod
    def compute_green_ratio(roi_bgr: np.ndarray) -> float:
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.countNonZero(mask) / max(mask.shape[0] * mask.shape[1], 1)

    # --------------------------------------------------------
    # Prediction scheduling
    # --------------------------------------------------------
    def toggle_auto(self) -> bool:
        with self.result_lock:
            self.auto_enabled = not self.auto_enabled
            enabled = self.auto_enabled
            self.last_auto_request_time = 0.0
        return enabled

    def request_manual_prediction(self) -> bool:
        with self.frame_lock:
            if self.latest_bgr is None:
                self.last_error = "Chưa có frame camera"
                return False
            frame = self.latest_bgr.copy()

        x1, y1, x2, y2 = self.compute_roi(frame.shape[1], frame.shape[0])
        roi = frame[y1:y2, x1:x2]
        return self.enqueue_prediction("MANUAL", roi)

    def maybe_request_auto_prediction(self, roi_bgr: np.ndarray) -> None:
        now = time.monotonic()

        with self.result_lock:
            enabled = self.auto_enabled
            busy = self.inference_busy
            last_request = self.last_auto_request_time

        if not enabled or busy:
            return

        if now - last_request < AUTO_PREDICT_INTERVAL_SEC:
            return

        if USE_HSV_PLANT_GATE and self.green_ratio < GREEN_RATIO_THRESHOLD:
            return

        if self.enqueue_prediction("AUTO", roi_bgr.copy()):
            with self.result_lock:
                self.last_auto_request_time = now

    def enqueue_prediction(self, trigger: str, roi_bgr: np.ndarray) -> bool:
        if self.model is None:
            self.last_error = "Mô hình chưa sẵn sàng"
            return False
        if roi_bgr is None or roi_bgr.size == 0:
            self.last_error = "ROI không hợp lệ"
            return False

        with self.result_lock:
            if self.inference_busy:
                self.last_error = "Mô hình đang dự đoán"
                return False

        try:
            self.inference_queue.put_nowait((trigger, roi_bgr.copy()))
            return True
        except queue.Full:
            self.last_error = "Hàng đợi dự đoán đang bận"
            return False

    # --------------------------------------------------------
    # Inference worker
    # --------------------------------------------------------
    def start_inference_worker(self) -> None:
        self.inference_thread = threading.Thread(
            target=self.inference_loop,
            name="growth_stage_inference",
            daemon=True,
        )
        self.inference_thread.start()

    def inference_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                trigger, image_bgr = self.inference_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            with self.result_lock:
                self.inference_busy = True
                self.last_trigger = trigger

            started = time.perf_counter()
            try:
                stage, raw_class, confidence = self.predict_growth_stage(image_bgr)
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                with self.result_lock:
                    self.predicted_stage = stage
                    self.predicted_class = raw_class
                    self.confidence = confidence
                    self.inference_ms = elapsed_ms
                    self.prediction_count += 1
                    self.last_error = "-"

            except Exception as exc:
                with self.result_lock:
                    self.predicted_stage = "Không xác định"
                    self.predicted_class = "-"
                    self.confidence = 0.0
                    self.inference_ms = (time.perf_counter() - started) * 1000.0
                    self.last_error = str(exc)
            finally:
                with self.result_lock:
                    self.inference_busy = False
                self.inference_queue.task_done()

    def predict_growth_stage(self, image_bgr: np.ndarray) -> tuple[str, str, float]:
        """Hỗ trợ cả model classification và model detection Ultralytics."""
        if self.model is None:
            raise RuntimeError("Mô hình chưa được nạp")

        results = self.model.predict(
            source=image_bgr,
            imgsz=MODEL_IMGSZ,
            device=MODEL_DEVICE,
            verbose=False,
        )
        if not results:
            raise RuntimeError("Mô hình không trả kết quả")

        result = results[0]
        names = result.names if getattr(result, "names", None) is not None else {}

        # Trường hợp image classification.
        probs = getattr(result, "probs", None)
        if probs is not None and probs.top1 is not None:
            class_id = int(probs.top1)
            confidence = float(probs.top1conf.item())
            raw_name = str(names.get(class_id, class_id))
            stage = self.map_stage_label(class_id, raw_name)
            return stage, raw_name, confidence

        # Trường hợp object detection: lấy bounding box tin cậy nhất.
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            confidences = boxes.conf.detach().cpu().numpy()
            best_index = int(np.argmax(confidences))
            confidence = float(confidences[best_index])
            class_id = int(boxes.cls[best_index].item())
            raw_name = str(names.get(class_id, class_id))
            stage = self.map_stage_label(class_id, raw_name)
            return stage, raw_name, confidence

        raise RuntimeError("Không phát hiện được giai đoạn sinh trưởng")

    @staticmethod
    def map_stage_label(class_id: int, raw_name: str) -> str:
        normalized = raw_name.strip().lower()

        if raw_name.strip() in NUMERIC_NAME_TO_LABEL:
            return NUMERIC_NAME_TO_LABEL[raw_name.strip()]

        aliases = {
            "seedling": "Cây con",
            "cay con": "Cây con",
            "cây con": "Cây con",
            "mature": "Trưởng thành",
            "adult": "Trưởng thành",
            "truong thanh": "Trưởng thành",
            "trưởng thành": "Trưởng thành",
            "flowering": "Ra hoa",
            "flower": "Ra hoa",
            "ra hoa": "Ra hoa",
            "fruit": "Có quả",
            "fruiting": "Có quả",
            "qua": "Có quả",
            "có quả": "Có quả",
            "ripe": "Quả chín",
            "ripe fruit": "Quả chín",
            "qua chin": "Quả chín",
            "quả chín": "Quả chín",
        }
        if normalized in aliases:
            return aliases[normalized]

        return STAGE_LABELS.get(class_id, raw_name)

    # --------------------------------------------------------
    # GUI snapshot / shutdown
    # --------------------------------------------------------
    def get_snapshot(self) -> dict:
        with self.frame_lock:
            frame = None if self.latest_display_bgr is None else self.latest_display_bgr.copy()
            green_ratio = self.green_ratio

        with self.result_lock:
            return {
                "frame": frame,
                "camera_status": self.camera_status,
                "model_status": self.model_status,
                "auto_enabled": self.auto_enabled,
                "busy": self.inference_busy,
                "stage": self.predicted_stage,
                "raw_class": self.predicted_class,
                "confidence": self.confidence,
                "inference_ms": self.inference_ms,
                "green_ratio": green_ratio,
                "trigger": self.last_trigger,
                "count": self.prediction_count,
                "error": self.last_error,
            }

    def shutdown(self) -> None:
        self.stop_event.set()

        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        if self.inference_thread is not None and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)

        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None


class GrowthMonitorGUI:
    """Giao diện Tkinter cổ điển, nhấn mạnh rõ 5 nhãn sinh trưởng."""

    STAGES = [
        ("1", "CÂY CON"),
        ("2", "TRƯỞNG THÀNH"),
        ("3", "RA HOA"),
        ("4", "CÓ QUẢ"),
        ("5", "QUẢ CHÍN"),
    ]

    def __init__(self, root: tk.Tk, monitor: GrowthMonitor) -> None:
        self.root = root
        self.monitor = monitor
        self.photo_image = None
        self.stage_cells: list[tk.Label] = []

        root.title("Hệ thống giám sát sinh trưởng cây trồng")
        root.geometry(f"{GUI_WIDTH}x{GUI_HEIGHT}+340+0")
        root.resizable(False, False)
        root.configure(bg="#d9d9d9")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Times New Roman", 15, "bold"))
        style.configure("Info.TLabel", font=("Times New Roman", 9))
        style.configure("Control.TButton", font=("Times New Roman", 10, "bold"))

        main = ttk.Frame(root, padding=7)
        main.pack(fill="both", expand=True)

        title = ttk.Label(
            main,
            text="HỆ THỐNG GIÁM SÁT GIAI ĐOẠN SINH TRƯỞNG CÂY TRỒNG",
            style="Title.TLabel",
            anchor="center",
        )
        title.pack(fill="x", pady=(0, 4))

        # Camera và cột nhãn nằm trên cùng một hàng.
        camera_stage_row = ttk.Frame(main)
        camera_stage_row.pack(fill="x")
        camera_stage_row.columnconfigure(0, weight=1)
        camera_stage_row.columnconfigure(1, weight=0)

        video_frame = ttk.Frame(camera_stage_row, padding=1)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.video_label = tk.Label(
            video_frame,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            bg="black",
            bd=1,
            relief="sunken",
        )
        self.video_label.pack()

        # Cột nhãn sinh trưởng đặt bên phải luồng camera.
        stages_frame = ttk.LabelFrame(
            camera_stage_row,
            text="Giai đoạn",
            padding=(4, 4),
            width=150,
            height=DISPLAY_HEIGHT,
        )
        stages_frame.grid(row=0, column=1, sticky="ns")
        stages_frame.grid_propagate(False)
        stages_frame.columnconfigure(0, weight=1)

        for row_index, (number, name) in enumerate(self.STAGES):
            stages_frame.rowconfigure(row_index, weight=1, uniform="stage")
            cell = tk.Label(
                stages_frame,
                text=f"{number}. {name}",
                font=("Times New Roman", 10, "bold"),
                bg="#e6e6e6",
                fg="#222222",
                relief="groove",
                bd=2,
                width=16,
                anchor="center",
                justify="center",
                wraplength=125,
            )
            cell.grid(
                row=row_index,
                column=0,
                sticky="nsew",
                padx=2,
                pady=2,
            )
            self.stage_cells.append(cell)

        control_frame = ttk.Frame(main)
        control_frame.pack(fill="x", pady=(4, 3))

        self.manual_button = ttk.Button(
            control_frame,
            text="MANUAL: DỰ ĐOÁN",
            command=self.on_manual,
            style="Control.TButton",
            width=20,
        )
        self.manual_button.pack(side="left", padx=(105, 10))

        self.auto_button = ttk.Button(
            control_frame,
            text="AUTO: TẮT",
            command=self.on_auto,
            style="Control.TButton",
            width=20,
        )
        self.auto_button.pack(side="left", padx=10)

        # Kết quả dự đoán chính.
        result_frame = ttk.LabelFrame(main, text="Kết quả dự đoán", padding=(5, 3))
        result_frame.pack(fill="x", pady=(1, 3))
        result_frame.columnconfigure(0, weight=1)
        result_frame.columnconfigure(1, weight=1)

        self.prediction_label = tk.Label(
            result_frame,
            text="CHƯA DỰ ĐOÁN",
            font=("Times New Roman", 15, "bold"),
            bg="#f2f2f2",
            fg="#202020",
            relief="ridge",
            bd=2,
            height=1,
            anchor="center",
        )
        self.prediction_label.grid(row=0, column=0, sticky="ew", padx=(2, 4))

        self.confidence_var = tk.StringVar(value="Độ tin cậy: 0.0% | 0 ms")
        ttk.Label(
            result_frame,
            textvariable=self.confidence_var,
            style="Info.TLabel",
            anchor="center",
        ).grid(row=0, column=1, sticky="ew", padx=(4, 2))

        status_frame = ttk.LabelFrame(main, text="Trạng thái hệ thống", padding=(5, 3))
        status_frame.pack(fill="x")

        self.status_var = tk.StringVar(value="Camera: khởi tạo | Mô hình: khởi tạo")
        self.detail_var = tk.StringVar(value="Chế độ: MANUAL | Số lần dự đoán: 0")
        self.error_var = tk.StringVar(value="Thông báo: -")

        ttk.Label(status_frame, textvariable=self.status_var, style="Info.TLabel", anchor="w").pack(fill="x")
        ttk.Label(status_frame, textvariable=self.detail_var, style="Info.TLabel", anchor="w").pack(fill="x")
        ttk.Label(status_frame, textvariable=self.error_var, style="Info.TLabel", anchor="w").pack(fill="x")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_gui()

    def on_manual(self) -> None:
        self.monitor.request_manual_prediction()

    def on_auto(self) -> None:
        enabled = self.monitor.toggle_auto()
        self.auto_button.configure(text="AUTO: BẬT" if enabled else "AUTO: TẮT")

    @staticmethod
    def normalize_stage_name(stage: str) -> str:
        return stage.strip().lower()

    def update_stage_highlight(self, stage: str) -> None:
        stage_normalized = self.normalize_stage_name(stage)
        lookup = {
            "cây con": 0,
            "trưởng thành": 1,
            "ra hoa": 2,
            "có quả": 3,
            "quả": 3,
            "quả chín": 4,
        }
        selected = lookup.get(stage_normalized)

        for index, cell in enumerate(self.stage_cells):
            if selected is not None and index == selected:
                cell.configure(
                    bg="#2e8b57",
                    fg="white",
                    relief="solid",
                    bd=3,
                )
            else:
                cell.configure(
                    bg="#e6e6e6",
                    fg="#222222",
                    relief="groove",
                    bd=2,
                )

    def refresh_gui(self) -> None:
        data = self.monitor.get_snapshot()
        frame = data["frame"]

        if frame is not None:
            resized = cv2.resize(
                frame,
                (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            self.photo_image = ImageTk.PhotoImage(image=pil_image)
            self.video_label.configure(image=self.photo_image)

        stage = data["stage"]
        busy_text = "ĐANG DỰ ĐOÁN..." if data["busy"] else stage.upper()
        self.prediction_label.configure(text=busy_text)

        if data["busy"]:
            self.prediction_label.configure(bg="#fff2cc", fg="#7f6000")
        elif stage not in {"Chưa dự đoán", "Không xác định"}:
            self.prediction_label.configure(bg="#d9ead3", fg="#1b5e20")
        else:
            self.prediction_label.configure(bg="#f2f2f2", fg="#202020")

        self.update_stage_highlight(stage)

        self.confidence_var.set(
            f"Độ tin cậy: {data['confidence'] * 100:.1f}%   |   "
            f"Thời gian suy luận: {data['inference_ms']:.0f} ms"
        )
        self.status_var.set(
            f"Camera: {data['camera_status']} | Mô hình: {data['model_status']} | "
            f"{'Đang xử lý' if data['busy'] else 'Sẵn sàng'}"
        )
        self.detail_var.set(
            f"Chế độ: {'AUTO' if data['auto_enabled'] else 'MANUAL'} | "
            f"Chu kỳ AUTO: {AUTO_PREDICT_INTERVAL_SEC:.1f} s | "
            f"Số lần dự đoán: {data['count']} | Kích hoạt: {data['trigger']}"
        )
        self.error_var.set(
            f"HSV xanh ROI: {data['green_ratio'] * 100:.1f}% | "
            # f"Nhãn gốc model: {data['raw_class']} | Thông báo: {data['error']}"
        )
        self.auto_button.configure(text="AUTO: BẬT" if data["auto_enabled"] else "AUTO: TẮT")

        self.root.after(50, self.refresh_gui)

    def on_close(self) -> None:
        self.root.destroy()

def main() -> None:
    monitor = GrowthMonitor()
    root = tk.Tk()
    GrowthMonitorGUI(root, monitor)

    try:
        root.mainloop()
    finally:
        monitor.shutdown()


if __name__ == "__main__":
    main()

