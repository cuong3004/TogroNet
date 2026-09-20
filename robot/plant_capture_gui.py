#!/usr/bin/env python3
"""
Plant Capture GUI for an autonomous robot using Raspberry Pi Camera V2 + ROS 2 Nav2.

Flow:
1. Read frames directly from Raspberry Pi Camera V2 using Picamera2/libcamera.
2. Analyze a fixed central ROI in HSV color space.
3. When vegetation is detected for several consecutive frames:
   - cancel the current NavigateToPose goal;
   - wait 1 second;
   - save the newest camera frame;
   - wait for cooldown;
   - send the same Nav2 goal again.
4. Display a classical Tkinter interface sized 684 x 600.

Required ROS packages:
  rclpy, nav2_msgs, action_msgs
Required Raspberry Pi OS packages:
  python3-picamera2, python3-opencv, python3-pil, python3-pil.imagetk, python3-tk
"""

from __future__ import annotations

import math
import re
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from PIL import Image, ImageTk
from picamera2 import Picamera2
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

import tkinter as tk
from tkinter import ttk


# ============================================================
# FIXED TECHNICAL PARAMETERS
# ============================================================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 20.0
CAMERA_WARMUP_SEC = 1.0
FRAME_ID = "map"

# Destination B. Change these values for your map.
GOAL_X = 2.0
GOAL_Y = 0.0
GOAL_YAW = 0.0

# Fixed HSV range for green vegetation in OpenCV scale:
# H: 0..179, S: 0..255, V: 0..255
HSV_LOWER = np.array([25, 45, 35], dtype=np.uint8)
HSV_UPPER = np.array([95, 255, 255], dtype=np.uint8)

# Central ROI as a fraction of the full image.
ROI_WIDTH_RATIO = 0.45
ROI_HEIGHT_RATIO = 0.55
ROI_CENTER_X_RATIO = 0.50
ROI_CENTER_Y_RATIO = 0.50

# Plant detection parameters.
GREEN_RATIO_THRESHOLD = 0.22
REQUIRED_POSITIVE_FRAMES = 5
REQUIRED_NEGATIVE_FRAMES_TO_REARM = 10

STOP_BEFORE_CAPTURE_SEC = 1.0
CAPTURE_COOLDOWN_SEC = 5.0

DATASET_ROOT = Path.home() / "dev_ws/dataset"

GUI_WIDTH = 684
GUI_HEIGHT = 600
DISPLAY_WIDTH = 560
DISPLAY_HEIGHT = 420


class PlantCaptureNode(Node):
    """ROS 2 node for image analysis, Nav2 control and image saving."""

    def __init__(self) -> None:
        super().__init__("plant_capture_gui")

        self.frame_lock = threading.Lock()
        self.camera_stop_event = threading.Event()
        self.camera_thread: Optional[threading.Thread] = None
        self.picam2: Optional[Picamera2] = None

        self.latest_bgr: Optional[np.ndarray] = None
        self.latest_display_bgr: Optional[np.ndarray] = None

        self.green_ratio = 0.0
        self.plant_visible = False
        self.capture_count = 0
        self.last_saved_file = "-"

        self.positive_frames = 0
        self.negative_frames = 0
        self.detector_armed = True
        self.auto_mode_enabled = False
        self.capture_trigger = "NONE"

        self.state = "WAITING FOR CAMERA"
        self.nav_status = "Waiting for Nav2"

        self.goal_handle = None
        self.goal_result_future = None
        self.cancel_future = None
        self.goal_active = False
        self.pause_requested = False

        self.capture_deadline = 0.0
        self.cooldown_deadline = 0.0

        self.start_camera()

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            "/navigate_to_pose",
        )

        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.server_timer = self.create_timer(1.0, self.check_nav2_server)

        self.get_logger().info("Plant Capture GUI node started")
        self.get_logger().info(
            f"Direct Pi Camera stream: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS:.0f} FPS"
        )
        self.get_logger().info(f"Dataset root: {DATASET_ROOT}")

    # --------------------------------------------------------
    # Direct Raspberry Pi Camera V2 processing
    # --------------------------------------------------------
    def start_camera(self) -> None:
        """Initialize Picamera2 and start a dedicated capture thread."""
        try:
            self.picam2 = Picamera2()
            video_config = self.picam2.create_video_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                    "format": "BGR888",
                },
                controls={
                    "FrameRate": CAMERA_FPS,
                },
                buffer_count=4,
            )
            self.picam2.configure(video_config)
            self.picam2.start()

            self.camera_stop_event.clear()
            self.camera_thread = threading.Thread(
                target=self.camera_loop,
                name="picamera2_capture",
                daemon=True,
            )
            self.camera_thread.start()
            self.state = "CAMERA STARTING"
        except Exception as exc:
            self.state = "CAMERA ERROR"
            self.nav_status = f"Picamera2: {exc}"
            self.get_logger().error(f"Cannot start Raspberry Pi Camera: {exc}")

    def camera_loop(self) -> None:
        """Continuously read frames directly from Picamera2."""
        time.sleep(CAMERA_WARMUP_SEC)

        frame_period = 1.0 / max(CAMERA_FPS, 1.0)

        while not self.camera_stop_event.is_set():
            loop_start = time.monotonic()

            try:
                if self.picam2 is None:
                    break

                rgb_frame = self.picam2.capture_array("main")
                if rgb_frame is None or rgb_frame.size == 0:
                    continue

                # Picamera2 RGB888 -> OpenCV BGR.
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                self.process_camera_frame(frame)

            except Exception as exc:
                self.state = "CAMERA ERROR"
                self.nav_status = f"Camera capture: {exc}"
                self.get_logger().error(f"Camera capture error: {exc}")
                time.sleep(0.5)

            elapsed = time.monotonic() - loop_start
            remaining = frame_period - elapsed
            if remaining > 0:
                self.camera_stop_event.wait(remaining)

    def process_camera_frame(self, frame: np.ndarray) -> None:
        """Analyze one BGR frame, update detection state and GUI image."""
        if frame is None or frame.size == 0:
            return

        x1, y1, x2, y2 = self.compute_roi(frame.shape[1], frame.shape[0])
        roi = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # Remove isolated noise and fill small gaps.
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        green_pixels = cv2.countNonZero(mask)
        roi_pixels = max(mask.shape[0] * mask.shape[1], 1)
        ratio = green_pixels / roi_pixels

        plant_now = ratio >= GREEN_RATIO_THRESHOLD
        self.update_detection_debounce(plant_now)

        display = frame.copy()
        roi_color = (0, 255, 255) if not plant_now else (0, 0, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), roi_color, 2)

        cv2.putText(
            display,
            f"ROI green: {ratio * 100:.1f}%",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "PLANT" if plant_now else "NO PLANT",
            (14, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            roi_color,
            2,
            cv2.LINE_AA,
        )

        with self.frame_lock:
            self.latest_bgr = frame.copy()
            self.latest_display_bgr = display
            self.green_ratio = ratio
            self.plant_visible = plant_now

        if self.state in {"WAITING FOR CAMERA", "CAMERA STARTING"}:
            self.state = "CAMERA ACTIVE"

    def stop_camera(self) -> None:
        """Stop the capture thread and release libcamera resources."""
        self.camera_stop_event.set()

        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)

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

    def compute_roi(self, width: int, height: int) -> tuple[int, int, int, int]:
        roi_w = int(width * ROI_WIDTH_RATIO)
        roi_h = int(height * ROI_HEIGHT_RATIO)

        cx = int(width * ROI_CENTER_X_RATIO)
        cy = int(height * ROI_CENTER_Y_RATIO)

        x1 = max(0, cx - roi_w // 2)
        y1 = max(0, cy - roi_h // 2)
        x2 = min(width, x1 + roi_w)
        y2 = min(height, y1 + roi_h)

        return x1, y1, x2, y2

    def update_detection_debounce(self, plant_now: bool) -> None:
        if plant_now:
            self.positive_frames += 1
            self.negative_frames = 0
        else:
            self.positive_frames = 0
            self.negative_frames += 1

            if self.negative_frames >= REQUIRED_NEGATIVE_FRAMES_TO_REARM:
                self.detector_armed = True

        if (
            self.auto_mode_enabled
            and self.detector_armed
            and self.positive_frames >= REQUIRED_POSITIVE_FRAMES
            and self.goal_active
            and self.state == "NAVIGATING"
        ):
            self.detector_armed = False
            self.positive_frames = 0
            self.request_navigation_pause(trigger="AUTO")

    # --------------------------------------------------------
    # Nav2 action control
    # --------------------------------------------------------
    def check_nav2_server(self) -> None:
        if self.goal_active or self.goal_handle is not None:
            return

        if self.state in {
            "STOPPING",
            "WAITING 1 SECOND",
            "CAPTURING",
            "COOLDOWN",
            "FINISHED",
        }:
            return

        if self.nav_client.server_is_ready():
            self.server_timer.cancel()
            self.send_navigation_goal()
        else:
            self.nav_status = "Waiting for /navigate_to_pose"

    def build_goal_pose(self) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = FRAME_ID
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(GOAL_X)
        pose.pose.position.y = float(GOAL_Y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(GOAL_YAW / 2.0)
        pose.pose.orientation.w = math.cos(GOAL_YAW / 2.0)
        return pose

    def send_navigation_goal(self) -> None:
        if not self.nav_client.server_is_ready():
            self.nav_status = "Nav2 action server unavailable"
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.build_goal_pose()

        self.state = "SENDING NAV2 GOAL"
        self.nav_status = f"Goal B: ({GOAL_X:.2f}, {GOAL_Y:.2f})"

        future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback,
        )
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.state = "NAVIGATION ERROR"
            self.nav_status = str(exc)
            return

        if not goal_handle.accepted:
            self.state = "GOAL REJECTED"
            self.nav_status = "Nav2 rejected destination B"
            return

        self.goal_handle = goal_handle
        self.goal_active = True
        self.pause_requested = False
        self.state = "NAVIGATING"
        self.nav_status = "Robot is moving"

        self.goal_result_future = goal_handle.get_result_async()
        self.goal_result_future.add_done_callback(self.navigation_result_callback)

    def navigation_feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        distance = getattr(feedback, "distance_remaining", None)
        if distance is not None:
            self.nav_status = f"Moving - remaining {distance:.2f} m"

    def request_navigation_pause(self, trigger: str = "AUTO") -> bool:
        """Cancel Nav2 before either an automatic or manual image capture."""
        if self.goal_handle is None or not self.goal_active:
            return False

        if self.state != "NAVIGATING":
            return False

        self.capture_trigger = trigger
        self.pause_requested = True
        self.state = "STOPPING"

        if trigger == "MANUAL":
            self.nav_status = "Manual capture - canceling Nav2"
        else:
            self.nav_status = "Plant detected - canceling Nav2"

        self.cancel_future = self.goal_handle.cancel_goal_async()
        self.cancel_future.add_done_callback(self.cancel_done_callback)
        return True

    def toggle_auto_mode(self) -> bool:
        """Enable or disable automatic HSV-based plant capture."""
        self.auto_mode_enabled = not self.auto_mode_enabled
        self.positive_frames = 0
        self.negative_frames = 0
        self.detector_armed = True

        if self.auto_mode_enabled:
            self.nav_status = "Automatic capture mode enabled"
            self.get_logger().info("AUTO mode enabled")
        else:
            self.nav_status = "Automatic capture mode disabled"
            self.get_logger().info("AUTO mode disabled")

        return self.auto_mode_enabled

    def request_manual_capture(self) -> bool:
        """Capture one image on demand."""
        busy_states = {
            "STOPPING",
            "WAITING 1 SECOND",
            "CAPTURING",
            "COOLDOWN",
            "RESUMING",
        }
        if self.state in busy_states:
            self.nav_status = "Capture sequence is already running"
            return False

        if self.goal_active and self.state == "NAVIGATING":
            return self.request_navigation_pause(trigger="MANUAL")

        saved = self.save_current_image()
        if saved:
            self.capture_count += 1
            self.capture_trigger = "MANUAL"
            self.nav_status = "Manual image saved"
            self.get_logger().info("Manual image capture completed")
            return True

        self.nav_status = "No valid camera frame for manual capture"
        return False

    def cancel_done_callback(self, future) -> None:
        try:
            response = future.result()
            canceled = len(response.goals_canceling) > 0
        except Exception as exc:
            self.state = "CANCEL ERROR"
            self.nav_status = str(exc)
            return

        if canceled:
            self.goal_active = False
            self.state = "WAITING 1 SECOND"
            self.nav_status = "Robot stopped for image capture"
            self.capture_deadline = time.monotonic() + STOP_BEFORE_CAPTURE_SEC
        else:
            self.state = "CANCEL REJECTED"
            self.nav_status = "Nav2 did not accept cancel request"

    def navigation_result_callback(self, future) -> None:
        try:
            wrapped_result = future.result()
            status = wrapped_result.status
        except Exception as exc:
            self.state = "NAVIGATION ERROR"
            self.nav_status = str(exc)
            return

        self.goal_handle = None
        self.goal_active = False

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.state = "FINISHED"
            self.nav_status = "Destination B reached"
            return

        if status == GoalStatus.STATUS_CANCELED and self.pause_requested:
            # Normal transition: capture and resume are handled by control_loop().
            return

        if status == GoalStatus.STATUS_ABORTED:
            self.state = "NAVIGATION FAILED"
            self.nav_status = "Nav2 aborted the goal"
        else:
            self.state = "NAVIGATION ENDED"
            self.nav_status = f"Nav2 status code: {status}"

    # --------------------------------------------------------
    # Mission state machine
    # --------------------------------------------------------
    def control_loop(self) -> None:
        now = time.monotonic()

        if self.state == "WAITING 1 SECOND" and now >= self.capture_deadline:
            self.state = "CAPTURING"
            saved = self.save_current_image()

            if saved:
                self.capture_count += 1
                self.state = "COOLDOWN"
                self.nav_status = "Image saved - waiting before resume"
                self.cooldown_deadline = now + CAPTURE_COOLDOWN_SEC
            else:
                self.state = "CAPTURE ERROR"
                self.nav_status = "No valid camera frame to save"

        elif self.state == "COOLDOWN" and now >= self.cooldown_deadline:
            self.pause_requested = False
            self.state = "RESUMING"
            self.nav_status = "Resending destination B"
            self.send_navigation_goal()

    # --------------------------------------------------------
    # Dataset storage
    # --------------------------------------------------------
    def save_current_image(self) -> bool:
        with self.frame_lock:
            if self.latest_bgr is None:
                return False
            frame = self.latest_bgr.copy()

        date_folder = DATASET_ROOT / datetime.now().strftime("%Y-%m-%d")
        date_folder.mkdir(parents=True, exist_ok=True)

        next_index = self.find_next_image_index(date_folder)
        output_path = date_folder / f"{next_index}.jpg"

        success = cv2.imwrite(str(output_path), frame)
        if success:
            self.last_saved_file = str(output_path)
            self.get_logger().info(f"Saved image: {output_path}")
        return bool(success)

    @staticmethod
    def find_next_image_index(folder: Path) -> int:
        highest = 0
        number_pattern = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)

        for path in folder.iterdir():
            match = number_pattern.match(path.name)
            if match:
                highest = max(highest, int(match.group(1)))

        return highest + 1

    def shutdown(self) -> None:
        """Release camera resources safely; may be called more than once."""
        if getattr(self, "_shutdown_started", False):
            return

        self._shutdown_started = True
        self.get_logger().info("Plant Capture shutdown requested")
        self.stop_camera()

    def get_gui_snapshot(self):
        with self.frame_lock:
            frame = None if self.latest_display_bgr is None else self.latest_display_bgr.copy()
            ratio = self.green_ratio
            visible = self.plant_visible

        return {
            "frame": frame,
            "ratio": ratio,
            "visible": visible,
            "state": self.state,
            "nav_status": self.nav_status,
            "count": self.capture_count,
            "last_file": self.last_saved_file,
            "auto_enabled": self.auto_mode_enabled,
            "capture_trigger": self.capture_trigger,
        }


class PlantCaptureGUI:
    """Classical academic-style Tkinter interface."""

    def __init__(self, root: tk.Tk, node: PlantCaptureNode) -> None:
        self.root = root
        self.node = node
        self.photo_image = None
        self.closing = False

        root.title("Automatic Plant Image Capture - Autonomous Robot")
        root.geometry(f"{GUI_WIDTH}x{GUI_HEIGHT}+340+0")
        root.resizable(False, False)

        root.configure(bg="#d9d9d9")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Times New Roman", 15, "bold"))
        style.configure("Info.TLabel", font=("Times New Roman", 10))
        style.configure("State.TLabel", font=("Times New Roman", 10, "bold"))

        self.main_frame = ttk.Frame(root, padding=8)
        self.main_frame.pack(fill="both", expand=True)

        self.title_label = ttk.Label(
            self.main_frame,
            text="HỆ THỐNG TỰ ĐỘNG THU NHẬN ẢNH CÂY TRỒNG",
            style="Title.TLabel",
            anchor="center",
        )
        self.title_label.pack(fill="x", pady=(0, 6))

        # Khung hiển thị camera, không có tiêu đề
        self.video_frame = ttk.Frame(
            self.main_frame,
            padding=4,
        )
        self.video_frame.pack(fill="x")

        self.video_label = tk.Label(
            self.video_frame,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            bg="black",
            bd=1,
            relief="sunken",
        )
        self.video_label.pack()
        

        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill="x", pady=(6, 0))

        self.manual_button = ttk.Button(
            self.control_frame,
            text="MANUAL",
            command=self.on_manual_capture,
            width=18,
        )
        self.manual_button.pack(side="left", padx=(160, 10))

        self.auto_button = ttk.Button(
            self.control_frame,
            text="AUTO: TẮT",
            command=self.on_auto_toggle,
            width=18,
        )
        self.auto_button.pack(side="left", padx=10)

        self.info_frame = ttk.LabelFrame(
            self.main_frame,
            text="System information",
            padding=6,
        )
        self.info_frame.pack(fill="x", pady=(6, 0))

        self.state_var = tk.StringVar(value="State: initializing")
        self.detect_var = tk.StringVar(value="Plant detection: no")
        self.ratio_var = tk.StringVar(value="Green ratio in ROI: 0.0%")
        self.count_var = tk.StringVar(value="Saved images: 0")
        self.file_var = tk.StringVar(value="Latest file: -")

        ttk.Label(self.info_frame, textvariable=self.state_var, style="State.TLabel").grid(
            row=0, column=0, sticky="w", padx=4, pady=2
        )
        ttk.Label(self.info_frame, textvariable=self.detect_var, style="Info.TLabel").grid(
            row=0, column=1, sticky="w", padx=12, pady=2
        )
        ttk.Label(self.info_frame, textvariable=self.ratio_var, style="Info.TLabel").grid(
            row=1, column=0, sticky="w", padx=4, pady=2
        )
        ttk.Label(self.info_frame, textvariable=self.count_var, style="Info.TLabel").grid(
            row=1, column=1, sticky="w", padx=12, pady=2
        )
        ttk.Label(self.info_frame, textvariable=self.file_var, style="Info.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_gui()

    def on_manual_capture(self) -> None:
        """MANUAL: capture one image on demand."""
        self.node.request_manual_capture()

    def on_auto_toggle(self) -> None:
        """AUTO: press once to enable, press again to disable."""
        enabled = self.node.toggle_auto_mode()
        self.auto_button.configure(
            text="AUTO: BẬT" if enabled else "AUTO: TẮT"
        )

    def refresh_gui(self) -> None:
        if self.closing:
            return

        data = self.node.get_gui_snapshot()
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

        self.state_var.set(f"State: {data['state']} | {data['nav_status']}")
        self.detect_var.set(
            "Plant detection: YES" if data["visible"] else "Plant detection: NO"
        )
        self.ratio_var.set(f"Green ratio in ROI: {data['ratio'] * 100:.1f}%")
        self.count_var.set(f"Saved images: {data['count']}")
        self.file_var.set(f"Latest file: {data['last_file']}")
        self.auto_button.configure(
            text="AUTO: BẬT" if data["auto_enabled"] else "AUTO: TẮT"
        )

        if not self.closing:
            self.root.after(50, self.refresh_gui)

    def on_close(self) -> None:
        """Request the Tk event loop to end; cleanup is done in main()."""
        if self.closing:
            return

        self.closing = True

        try:
            self.root.quit()
        except tk.TclError:
            pass


def main(args=None) -> None:
    rclpy.init(args=args)

    node = PlantCaptureNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    spin_thread = threading.Thread(
        target=executor.spin,
        name="plant_capture_ros_executor",
        daemon=True,
    )
    spin_thread.start()

    root = tk.Tk()
    gui = PlantCaptureGUI(root, node)

    shutdown_requested = threading.Event()

    def request_shutdown(signum=None, frame=None) -> None:
        """
        Handle systemd SIGTERM, Ctrl+C and the window close button gracefully.

        Tkinter calls must run in the Tk main thread, so the signal handler
        only schedules gui.on_close().
        """
        if shutdown_requested.is_set():
            return

        shutdown_requested.set()

        try:
            root.after(0, gui.on_close)
        except tk.TclError:
            pass

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    try:
        root.mainloop()
    finally:
        # Prevent GUI callbacks from scheduling new work.
        gui.closing = True

        # Release libcamera before the process exits. This avoids the camera
        # remaining busy when the service is started again.
        node.shutdown()

        try:
            executor.remove_node(node)
        except Exception:
            pass

        try:
            executor.shutdown(timeout_sec=3.0)
        except TypeError:
            # Compatibility with rclpy versions without timeout_sec.
            executor.shutdown()
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

        spin_thread.join(timeout=3.0)

        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
