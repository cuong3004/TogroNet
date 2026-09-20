#!/usr/bin/env python3

import tkinter as tk
from tkinter import messagebox
import subprocess
import threading
import shlex
import re
import time
import os
import time

# =========================
# CẤU HÌNH CHÍNH
# =========================

PI3_HOST = "pi@192.168.50.3"

# File RViz trên Pi4
RVIZ_DEFAULT = "/home/pi/dev_ws/src/my_robot_bringup/rviz/default.rviz"
RVIZ_SLAM    = "/home/pi/dev_ws/src/my_robot_bringup/rviz/slam.rviz"
RVIZ_NAV     = "/home/pi/dev_ws/src/my_robot_bringup/rviz/nav.rviz"

# Nếu service của bạn tên có .service cũng được,
# nhưng systemctl thường chỉ cần tên ngắn.
PI3_ROBOT_SERVICE = "robot"

PI4_LIDAR_SERVICE = "lidar"
PI4_SLAM_SERVICE = "slam"
PI4_NAV_SERVICE = "nav"
PI4_TOMATO_SERVICE = "tomato"

PI4_PLANT_CAPTURE_SERVICE = "plant-capture"
PI4_GROWTH_MONITOR_SERVICE = "plant-growth-monitor"

# =========================
# NAV2 ACTION
# =========================
SPIN_ACTION = "/spin"
SPIN_ANGLE_RAD = 6.283185307179586   # 360 độ = 2*pi rad
SPIN_TIMEOUT_SEC = 30


LOGO_LEFT = "/home/pi/dev_ws/logo_hus.png"
LOGO_RIGHT = "/home/pi/dev_ws/logo_tnut.png"
ROBOT_IMAGE = "/home/pi/dev_ws/robot.png"


import rclpy

from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


# =========================
# TELEOP KEYBOARD (giống hệt gói teleop_twist_keyboard)
# =========================
TELEOP_MOVE_BINDINGS = {
    'i': (1, 0, 0, 0),
    'o': (1, 0, 0, -1),
    'j': (0, 0, 0, 1),
    'l': (0, 0, 0, -1),
    'u': (1, 0, 0, 1),
    ',': (-1, 0, 0, 0),
    '.': (-1, 0, 0, 1),
    'm': (-1, 0, 0, -1),
    'O': (1, -1, 0, 0),
    'I': (1, 0, 0, 0),
    'J': (0, 1, 0, 0),
    'L': (0, -1, 0, 0),
    'U': (1, 1, 0, 0),
    '<': (-1, 0, 0, 0),
    '>': (-1, -1, 0, 0),
    'M': (-1, 1, 0, 0),
    't': (0, 0, 1, 0),
    'b': (0, 0, -1, 0),
}

TELEOP_SPEED_BINDINGS = {
    'q': (1.1, 1.1),
    'z': (.9, .9),
    'w': (1.1, 1),
    'x': (.9, 1),
    'e': (1, 1.1),
    'c': (1, .9),
}


class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__("robot_gui_cmd_vel")

        self.publisher = self.create_publisher(
            TwistStamped,
            "/diff_drive_controller/cmd_vel",
            10,
        )

    def send_velocity(self, linear_x=0.0, linear_y=0.0, linear_z=0.0, angular_z=0.0):
        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""

        msg.twist.linear.x = float(linear_x)
        msg.twist.linear.y = float(linear_y)
        msg.twist.linear.z = float(linear_z)

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = float(angular_z)

        self.publisher.publish(msg)


# =========================
# HÀM CHẠY LỆNH
# =========================

def run_cmd(cmd, timeout=15):
    """
    Chạy lệnh shell và trả về (returncode, stdout, stderr).
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "Command timeout"


def local_systemctl(action, service):
    """
    Chạy systemctl trên Pi4.
    """
    cmd = f"sudo systemctl {action} {shlex.quote(service)}"
    return run_cmd(cmd)


def remote_pi3_systemctl(action, service):
    """
    Chạy systemctl trên Pi3 qua SSH.
    """
    remote_cmd = f"sudo systemctl {action} {shlex.quote(service)}"
    cmd = f"ssh {shlex.quote(PI3_HOST)} {shlex.quote(remote_cmd)}"
    return run_cmd(cmd)


def local_is_active(service):
    rc, out, err = local_systemctl("is-active", service)
    if out:
        return out
    return "unknown"


def remote_pi3_is_active(service):
    rc, out, err = remote_pi3_systemctl("is-active", service)
    if out:
        return out
    return "unknown"


# =========================
# RVIZ TEMPLATE SERVICE
# =========================

current_rviz_instance = None


def systemd_escape_path(path):
    """
    Escape đường dẫn để dùng với rviz@.service.
    """
    cmd = f"systemd-escape --path {shlex.quote(path)}"
    rc, out, err = run_cmd(cmd)
    if rc != 0 or not out:
        raise RuntimeError(f"systemd-escape failed: {err}")
    return out


def start_rviz_with_file(rviz_file):
    """
    Stop RViz hiện tại, rồi start rviz@<escaped_path>.service.
    """
    global current_rviz_instance

    stop_current_rviz()

    instance = systemd_escape_path(rviz_file)
    service = f"rviz@{instance}.service"

    rc, out, err = local_systemctl("start", service)
    if rc != 0:
        raise RuntimeError(err or out or f"Failed to start {service}")

    current_rviz_instance = service


def stop_current_rviz():
    """
    Dừng RViz mà GUI đã start.
    """
    global current_rviz_instance

    if current_rviz_instance:
        local_systemctl("stop", current_rviz_instance)
        current_rviz_instance = None
        return

    # Nếu GUI không biết RViz instance nào đang chạy,
    # thử stop toàn bộ rviz@*.service đang active.
    rc, out, err = run_cmd("systemctl list-units 'rviz@*.service' --no-legend --plain")
    if out:
        for line in out.splitlines():
            parts = line.split()
            if parts:
                service_name = parts[0]
                local_systemctl("stop", service_name)


# =========================
# MAP NAME
# =========================

def clean_map_name(raw):
    """
    Chỉ cho phép tên map đơn giản để tránh lỗi shell/systemd.
    Ví dụ hợp lệ:
      tomato_lab
      greenhouse_01
      map_2026_07_08
    """
    name = raw.strip()

    if not name:
        raise ValueError("Bạn chưa nhập tên map.")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(
            "Tên map chỉ nên gồm chữ, số, dấu gạch dưới hoặc gạch ngang.\n"
            "Ví dụ: tomato_lab, greenhouse_01, map_2026_07_08"
        )

    return name


# =========================
# GUI APP
# =========================

class RobotControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Control Panel")
        # self.root.geometry("430x280")
        self.root.geometry("340x600+0+0")
        self.root.resizable(False, False)

        self.busy = False

        # =========================
        # TELEOP KEYBOARD STATE
        # =========================
        self.teleop_speed = 0.10
        self.teleop_turn = 0.50
        self.teleop_x = 0.0
        self.teleop_y = 0.0
        self.teleop_z = 0.0
        self.teleop_th = 0.0

        # Phím di chuyển đang được giữ (None = không phím nào)
        self.teleop_active_key = None
        # id của root.after() đang lặp publish, hoặc đang chờ xác nhận thả phím
        self.teleop_publish_job = None
        self.teleop_release_job = None

        # Bộ diff_drive_controller có cmd_vel_timeout: 0.5s -> phải publish
        # lặp lại nhanh hơn mức đó trong lúc giữ phím, không thể chỉ gửi
        # đúng 1 lần mỗi lần nhấn (robot sẽ tự dừng sau 0.5s rồi giật lại).
        self.TELEOP_PUBLISH_PERIOD_MS = 50   # 20 Hz
        # X11 tự phát sinh cặp Release/Press giả khi giữ phím (auto-repeat).
        # Trì hoãn xử lý Release một chút để phát hiện và bỏ qua cặp giả đó.
        self.TELEOP_RELEASE_DEBOUNCE_MS = 80

        self.cmd_vel_node = CmdVelPublisher()

        self.colors = {
            "inactive": "#6b7280",   # xám
            "active": "#16a34a",     # xanh lá
            "partial": "#f59e0b",    # cam
            "failed": "#dc2626",     # đỏ
            "busy": "#2563eb",       # xanh dương
            "unknown": "#7c3aed",    # tím
        }

        self.build_ui()
        self.bind_teleop_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_status_loop()

    # -------------------------
    # TELEOP KEYBOARD (giống hệt gói teleop_twist_keyboard)
    # -------------------------

    def bind_teleop_keys(self):
        """
        Bắt phím trên toàn bộ cửa sổ. Bỏ qua khi đang gõ vào ô
        nhập tên map để không gõ chữ mà robot lại chạy.
        """
        self.root.bind_all("<KeyPress>", self.on_teleop_key_press)
        self.root.bind_all("<KeyRelease>", self.on_teleop_key_release)

        # Click ra ngoài ô nhập tên map -> nhả con trỏ, để phím quay
        # lại điều khiển robot ngay (trước đây con trỏ đứng yên mãi
        # trong ô Entry sau khi gõ xong, không bấm phím di chuyển được).
        self.root.bind_all("<Button-1>", self.on_click_maybe_release_entry)

    def on_click_maybe_release_entry(self, event):
        if event.widget is not self.map_entry:
            self.root.focus_set()

    def on_teleop_key_press(self, event):
        if event.widget is self.map_entry:
            return

        key = event.char

        if key in TELEOP_MOVE_BINDINGS:
            # Giữ phím cũ đang lặp -> huỷ lệnh dừng trì hoãn (đây là auto-repeat).
            if key == self.teleop_active_key and self.teleop_release_job is not None:
                self.root.after_cancel(self.teleop_release_job)
                self.teleop_release_job = None
                return

            self.teleop_active_key = key
            self.teleop_x, self.teleop_y, self.teleop_z, self.teleop_th = (
                TELEOP_MOVE_BINDINGS[key]
            )
            self.start_teleop_repeat()
        elif key in TELEOP_SPEED_BINDINGS:
            speed_ratio, turn_ratio = TELEOP_SPEED_BINDINGS[key]
            self.teleop_speed *= speed_ratio
            self.teleop_turn *= turn_ratio
            self.publish_teleop_velocity()
        else:
            self.stop_teleop_motion()

    def on_teleop_key_release(self, event):
        if event.widget is self.map_entry:
            return

        key = event.char

        if key != self.teleop_active_key:
            return

        # Không dừng ngay: chờ một nhịp ngắn xem có phải auto-repeat
        # (Press giả sẽ đến ngay sau) hay là thả phím thật.
        self.teleop_release_job = self.root.after(
            self.TELEOP_RELEASE_DEBOUNCE_MS,
            self.stop_teleop_motion,
        )

    def start_teleop_repeat(self):
        if self.teleop_publish_job is not None:
            self.root.after_cancel(self.teleop_publish_job)

        def tick():
            self.publish_teleop_velocity()
            self.teleop_publish_job = self.root.after(
                self.TELEOP_PUBLISH_PERIOD_MS,
                tick,
            )

        tick()

    def stop_teleop_motion(self):
        if self.teleop_publish_job is not None:
            self.root.after_cancel(self.teleop_publish_job)
            self.teleop_publish_job = None

        if self.teleop_release_job is not None:
            self.root.after_cancel(self.teleop_release_job)
            self.teleop_release_job = None

        self.teleop_active_key = None
        self.teleop_x = 0.0
        self.teleop_y = 0.0
        self.teleop_z = 0.0
        self.teleop_th = 0.0

        self.publish_teleop_velocity()

    def publish_teleop_velocity(self):
        self.cmd_vel_node.send_velocity(
            linear_x=self.teleop_x * self.teleop_speed,
            linear_y=self.teleop_y * self.teleop_speed,
            linear_z=self.teleop_z * self.teleop_speed,
            angular_z=self.teleop_th * self.teleop_turn,
        )

    def on_close(self):
        try:
            self.stop_teleop_motion()
        except Exception:
            pass

        try:
            self.cmd_vel_node.destroy_node()
        except Exception:
            pass

        try:
            rclpy.shutdown()
        except Exception:
            pass

        self.root.destroy()

    def build_ui(self):
        # =========================
        # CỬA SỔ CHÍNH
        # =========================
        self.root.geometry("340x600+0+0")
        self.root.resizable(False, False)
        self.root.configure(bg="#eef2f5")

        # Giữ cửa sổ sát trái trên khi mở
        self.root.update_idletasks()
        self.root.geometry("340x600+0+0")

        # =========================
        # ẢNH / LOGO
        # =========================
        self.logo_left_img = self.load_image(LOGO_LEFT, (52, 52))
        self.logo_right_img = self.load_image(LOGO_RIGHT, (52, 52))
        self.robot_img = self.load_image(ROBOT_IMAGE, (170, 123))

        # =========================
        # HEADER
        # =========================
        header = tk.Frame(self.root, bg="#eef2f5")
        header.pack(fill="x", padx=8, pady=(6, 2))

        left_logo_box = tk.Frame(header, width=58, height=58, bg="#eef2f5")
        left_logo_box.pack(side="left")
        left_logo_box.pack_propagate(False)

        if self.logo_left_img:
            tk.Label(left_logo_box, image=self.logo_left_img, bg="#eef2f5").pack(expand=True)
        else:
            tk.Label(left_logo_box, text="LOGO", bg="#d9dee3", font=("Arial", 8)).pack(expand=True, fill="both")

        title_box = tk.Frame(header, bg="#eef2f5")
        title_box.pack(side="left", expand=True, fill="both")

        tk.Label(
            title_box,
            text="ROBOT",
            font=("Arial", 14, "bold"),
            fg="#1f2933",
            bg="#eef2f5"
        ).pack(pady=(3, 0))

        tk.Label(
            title_box,
            text="Control Panel",
            font=("Arial", 10),
            fg="#52616b",
            bg="#eef2f5"
        ).pack()

        right_logo_box = tk.Frame(header, width=58, height=58, bg="#eef2f5")
        right_logo_box.pack(side="right")
        right_logo_box.pack_propagate(False)

        if self.logo_right_img:
            tk.Label(right_logo_box, image=self.logo_right_img, bg="#eef2f5").pack(expand=True)
        else:
            tk.Label(right_logo_box, text="LOGO", bg="#d9dee3", font=("Arial", 8)).pack(expand=True, fill="both")

        # Đường phân cách
        tk.Frame(self.root, height=1, bg="#c9d1d9").pack(fill="x", padx=10, pady=(4, 6))

        # =========================
        # ẢNH ROBOT
        # =========================
        robot_box = tk.Frame(self.root, bg="#eef2f5")
        robot_box.pack(fill="x", padx=8, pady=(0, 4))

        if self.robot_img:
            tk.Label(robot_box, image=self.robot_img, bg="#eef2f5").pack()
        else:
            tk.Label(
                robot_box,
                text="AUTONOMOUS ROBOT",
                bg="#d9dee3",
                fg="#333333",
                font=("Arial", 10, "bold"),
                height=3
            ).pack(fill="x", padx=20)

        # =========================
        # MAP INPUT
        # =========================
        map_frame = tk.LabelFrame(
            self.root,
            text=" Map ",
            font=("Arial", 9, "bold"),
            bg="#eef2f5",
            fg="#1f2933",
            padx=8,
            pady=6
        )
        map_frame.pack(fill="x", padx=10, pady=(4, 6))

        self.map_entry = tk.Entry(
            map_frame,
            width=26,
            font=("Arial", 12),
            justify="center",
            relief="solid",
            bd=1
        )
        self.map_entry.pack(fill="x")
        self.map_entry.insert(0, "tomato_lab")

        # =========================
        # BUTTON AREA
        # =========================
        button_frame = tk.Frame(self.root, bg="#eef2f5")
        button_frame.pack(fill="x", padx=10, pady=(4, 4))

        def make_button(parent, text, command):
            btn = tk.Button(
                parent,
                text=text,
                width=13,
                height=2,
                command=command,
                font=("Arial", 10, "bold"),
                fg="white",
                bg="#777777",
                activebackground="#555555",
                activeforeground="white",
                relief="raised",
                bd=2,
                cursor="hand2"
            )
            return btn

        # Hàng 1
        self.btn_robot = make_button(button_frame, "ROBOT", self.on_robot)
        self.btn_robot.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.btn_slam = make_button(button_frame, "SLAM", self.on_slam)
        self.btn_slam.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Hàng 2
        self.btn_savemap = make_button(button_frame, "SAVE MAP", self.on_savemap)
        self.btn_savemap.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.btn_loadmap = make_button(button_frame, "LOAD MAP", self.on_loadmap)
        self.btn_loadmap.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # Hàng 3
        self.btn_nav = make_button(button_frame, "NAV", self.on_nav)
        self.btn_nav.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        self.btn_tomato = make_button(button_frame, "TOMATO", self.on_tomato)
        self.btn_tomato.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")

        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        # =========================
        # LEGEND MÀU
        # =========================
        legend = tk.Frame(self.root, bg="#eef2f5")
        legend.pack(fill="x", padx=12, pady=(2, 3))

        # def legend_item(parent, color, text):
        #     item = tk.Frame(parent, bg="#eef2f5")
        #     item.pack(side="left", padx=4)

        #     tk.Label(
        #         item,
        #         width=2,
        #         height=1,
        #         bg=color,
        #         relief="solid",
        #         bd=1
        #     ).pack(side="left")

        #     tk.Label(
        #         item,
        #         text=text,
        #         font=("Arial", 8),
        #         bg="#eef2f5",
        #         fg="#333333"
        #     ).pack(side="left", padx=(2, 0))

        # legend_item(legend, self.colors["inactive"], "Off")
        # legend_item(legend, self.colors["active"], "Run")
        # legend_item(legend, self.colors["partial"], "Part")
        # legend_item(legend, self.colors["failed"], "Fail")

        # =====================================================
        # Các nút phụ bên phải:
        # [SPIN] [CAPTURE] [GROWTH]
        #
        # Vì dùng pack(side="right"), nút pack trước sẽ nằm
        # ngoài cùng bên phải.
        # =====================================================

        # GROWTH nằm ngoài cùng bên phải
        self.growth_button = tk.Button(
            legend,
            text="GROWTH",
            font=("Arial", 8, "bold"),
            bg="#6b7280",
            fg="white",
            activebackground="#4b5563",
            activeforeground="white",
            disabledforeground="#d1d5db",
            relief="raised",
            bd=1,
            padx=6,
            pady=2,
            cursor="hand2",
            command=self.on_growth
        )

        self.growth_button.pack(
            side="right",
            padx=(4, 0)
        )

        # CAPTURE nằm giữa SPIN và GROWTH
        self.capture_button = tk.Button(
            legend,
            text="CAPTURE",
            font=("Arial", 8, "bold"),
            bg="#6b7280",
            fg="white",
            activebackground="#4b5563",
            activeforeground="white",
            disabledforeground="#d1d5db",
            relief="raised",
            bd=1,
            padx=6,
            pady=2,
            cursor="hand2",
            command=self.on_capture
        )

        self.capture_button.pack(
            side="right",
            padx=(4, 0)
        )

        # SPIN nằm bên trái CAPTURE
        self.spin_button = tk.Button(
            legend,
            text="⟳ SPIN",
            font=("Arial", 8, "bold"),
            bg="#1976d2",
            fg="white",
            activebackground="#1565c0",
            activeforeground="white",
            disabledforeground="#d1d5db",
            relief="raised",
            bd=1,
            padx=6,
            pady=2,
            cursor="hand2",
            command=self.spin_robot
        )

        self.spin_button.pack(
            side="right",
            padx=(4, 0)
        )

        # =========================
        # STATUS
        # =========================
        status_frame = tk.Frame(self.root, bg="#eef2f5")
        status_frame.pack(fill="both", expand=True, padx=10, pady=(2, 6))

        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            font=("Arial", 8),
            fg="#1f2933",
            bg="#ffffff",
            anchor="nw",
            justify="left",
            relief="solid",
            bd=1,
            padx=5,
            pady=4,
            wraplength=300
        )
        self.status_label.pack(fill="both", expand=True)

    # -------------------------
    # Helper cho GUI
    # -------------------------

    def load_image(self, path, size):
        """
        Load ảnh PNG/JPG, resize về kích thước mong muốn.
        Nếu thiếu ảnh thì trả về None để GUI vẫn chạy.
        """
        try:
            from PIL import Image, ImageTk

            img = Image.open(path)
            img = img.resize(size, Image.LANCZOS)
            return ImageTk.PhotoImage(img)

        except Exception as e:
            print(f"Cannot load image {path}: {e}")
            return None

    def set_button_state(self, button, state_name):
        color = self.colors.get(state_name, self.colors["unknown"])

        text_color = "white"

        button.config(
            bg=color,
            fg=text_color,
            activebackground=color,
            activeforeground=text_color
        )

    def set_busy_buttons(self, busy=True):
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in [
            self.btn_robot,
            self.btn_slam,
            self.btn_savemap,
            self.btn_loadmap,
            self.btn_nav,
            self.btn_tomato,
            self.spin_button,
            self.capture_button,
            self.growth_button,
        ]:
            btn.config(state=state)

    def run_background(self, task, done_msg=None):
        if self.busy:
            return

        self.busy = True
        self.set_busy_buttons(True)
        self.status_label.config(text="Running command...")

        def worker():
            try:
                task()
                if done_msg:
                    self.root.after(0, lambda: self.status_label.config(text=done_msg))
            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
                self.root.after(0, lambda: self.status_label.config(text="Error"))
            finally:
                self.busy = False
                self.root.after(0, lambda: self.set_busy_buttons(False))
                self.root.after(0, self.refresh_status_once)

        threading.Thread(target=worker, daemon=True).start()

    def check_group_state(self, states):
        """
        Gộp trạng thái nhiều service:
        - có failed -> failed
        - tất cả active -> active
        - tất cả inactive/unknown -> inactive
        - còn lại -> partial
        """
        if any(s == "failed" for s in states):
            return "failed"

        if states and all(s == "active" for s in states):
            return "active"

        if all(s in ["inactive", "unknown", "deactivating"] for s in states):
            return "inactive"

        return "partial"

    # -------------------------
    # REFRESH STATUS
    # -------------------------

    def refresh_status_once(self):
        def task():
            # Robot group: robot Pi3 + lidar Pi4
            robot_state = remote_pi3_is_active(PI3_ROBOT_SERVICE)
            lidar_state = local_is_active(PI4_LIDAR_SERVICE)
            robot_group = self.check_group_state([robot_state, lidar_state])

            slam_state = local_is_active(PI4_SLAM_SERVICE)
            nav_state = local_is_active(PI4_NAV_SERVICE)
            tomato_state = local_is_active(PI4_TOMATO_SERVICE)

            capture_state = local_is_active(PI4_PLANT_CAPTURE_SERVICE)
            growth_state = local_is_active(PI4_GROWTH_MONITOR_SERVICE)

            # Loadmap cần tên map hiện tại
            try:
                map_name = clean_map_name(self.map_entry.get())
                loadmap_state = local_is_active(f"loadmap@{map_name}.service")
            except Exception:
                loadmap_state = "inactive"

            def update_ui():
                self.set_button_state(self.btn_robot, robot_group)
                self.set_button_state(self.btn_slam, slam_state)
                self.set_button_state(self.btn_loadmap, loadmap_state)
                self.set_button_state(self.btn_nav, nav_state)
                self.set_button_state(self.btn_tomato, tomato_state)

                self.set_button_state(self.capture_button, capture_state)
                self.set_button_state(self.growth_button, growth_state)

                # Save map là oneshot, không có trạng thái chạy lâu dài
                self.set_button_state(self.btn_savemap, "inactive")

                self.status_label.config(
                    text=(
                        f"Robot:{robot_state}, "
                        f"Lidar:{lidar_state}, "
                        f"SLAM:{slam_state}, "
                        f"LoadMap:{loadmap_state}, "
                        f"Nav:{nav_state}, "
                        f"Tomato:{tomato_state}\n"
                        f"Capture:{capture_state}, "
                        f"Growth:{growth_state}"
                    )
                )

            self.root.after(0, update_ui)

        threading.Thread(target=task, daemon=True).start()

    def refresh_status_loop(self):
        if not self.busy:
            self.refresh_status_once()
        self.root.after(3000, self.refresh_status_loop)

    

    def on_capture(self):
        """
        Bật/tắt riêng plant-capture.service.
        """

        def task():
            state = local_is_active(
                PI4_PLANT_CAPTURE_SERVICE
            )

            if state == "inactive":
                stop_current_rviz()
                rc, out, err = local_systemctl(
                    "start",
                    PI4_PLANT_CAPTURE_SERVICE
                )

                if rc != 0:
                    raise RuntimeError(
                        "Start plant-capture failed:\n"
                        f"{err or out}"
                    )
            else:
                rc, out, err = local_systemctl(
                    "stop",
                    PI4_PLANT_CAPTURE_SERVICE
                )

                if rc != 0:
                    raise RuntimeError(
                        "Stop plant-capture failed:\n"
                        f"{err or out}"
                    )

        self.run_background(
            task,
            "Plant Capture toggled"
        )

    def on_growth(self):
        """
        Bat/tat rieng plant-growth-monitor.service (gio chay
        growth_summary_gui.py - xem file do de biet chi tiet noi dung).
        """

        def task():
            state = local_is_active(
                PI4_GROWTH_MONITOR_SERVICE
            )

            if state == "inactive":
                stop_current_rviz()
                rc, out, err = local_systemctl(
                    "start",
                    PI4_GROWTH_MONITOR_SERVICE
                )

                if rc != 0:
                    raise RuntimeError(
                        "Start plant-growth-monitor failed:\n"
                        f"{err or out}"
                    )
            else:
                rc, out, err = local_systemctl(
                    "stop",
                    PI4_GROWTH_MONITOR_SERVICE
                )

                if rc != 0:
                    raise RuntimeError(
                        "Stop plant-growth-monitor failed:\n"
                        f"{err or out}"
                    )

        self.run_background(
            task,
            "Growth Monitor toggled"
        )
    # -------------------------
    # BUTTON ACTIONS
    # -------------------------
    def spin_robot(self):
        """
        Chạy tuần tự 4 chuyển động:

        1. Tiến 2 giây
        2. Lùi 2 giây
        3. Quay trái 2 giây
        4. Quay phải 2 giây

        Gửi geometry_msgs/msg/TwistStamped vào:
        /diff_drive_controller/cmd_vel
        """

        CMD_VEL_TOPIC = "/diff_drive_controller/cmd_vel"
        CMD_VEL_TYPE = "geometry_msgs/msg/TwistStamped"

        PUBLISH_RATE = 10
        DURATION = 5.0

        LINEAR_SPEED = 0.05
        ANGULAR_SPEED = 0.50

        def task():

            def publish_velocity(linear_x, angular_z, duration, status_text):
                """
                Gửi vận tốc ở tần số PUBLISH_RATE trong duration giây.
                """

                self.root.after(
                    0,
                    lambda text=status_text: self.status_label.config(text=text)
                )

                message = (
                    "{"
                    "header: {stamp: now, frame_id: ''}, "
                    "twist: {"
                    f"linear: {{x: {linear_x}, y: 0.0, z: 0.0}}, "
                    f"angular: {{x: 0.0, y: 0.0, z: {angular_z}}}"
                    "}"
                    "}"
                )

                number_of_messages = max(
                    1,
                    int(duration * PUBLISH_RATE)
                )

                cmd = (
                    "ros2 topic pub "
                    f"--rate {PUBLISH_RATE} "
                    f"--times {number_of_messages} "
                    f"{shlex.quote(CMD_VEL_TOPIC)} "
                    f"{shlex.quote(CMD_VEL_TYPE)} "
                    f"{shlex.quote(message)}"
                )

                rc, out, err = run_cmd(
                    cmd,
                    timeout=duration +20
                )

                if rc != 0:
                    raise RuntimeError(
                        f"{status_text} thất bại:\n"
                        f"{err or out or 'Không có phản hồi.'}"
                    )

            def stop_robot():
                """
                Gửi vận tốc 0 để bảo đảm robot dừng.
                """

                stop_message = (
                    "{"
                    "header: {stamp: now, frame_id: ''}, "
                    "twist: {"
                    "linear: {x: 0.0, y: 0.0, z: 0.0}, "
                    "angular: {x: 0.0, y: 0.0, z: 0.0}"
                    "}"
                    "}"
                )

                stop_cmd = (
                    "ros2 topic pub "
                    "--once "
                    f"{shlex.quote(CMD_VEL_TOPIC)} "
                    f"{shlex.quote(CMD_VEL_TYPE)} "
                    f"{shlex.quote(stop_message)}"
                )

                run_cmd(stop_cmd, timeout=10)

            try:
                # 1. Tiến
                publish_velocity(
                    linear_x=LINEAR_SPEED,
                    angular_z=0.0,
                    duration=DURATION,
                    status_text="Robot đang tiến trong 2 giây..."
                )

                stop_robot()

                # 2. Lùi
                publish_velocity(
                    linear_x=-LINEAR_SPEED,
                    angular_z=0.0,
                    duration=DURATION,
                    status_text="Robot đang lùi trong 2 giây..."
                )

                stop_robot()

                # 3. Quay trái
                publish_velocity(
                    linear_x=0.0,
                    angular_z=ANGULAR_SPEED,
                    duration=DURATION,
                    status_text="Robot đang quay trái trong 2 giây..."
                )

                stop_robot()

                # 4. Quay phải
                publish_velocity(
                    linear_x=0.0,
                    angular_z=-ANGULAR_SPEED,
                    duration=DURATION,
                    status_text="Robot đang quay phải trong 2 giây..."
                )

                stop_robot()

            finally:
                # Luôn gửi lệnh dừng, kể cả khi giữa chừng xảy ra lỗi
                stop_robot()

        def restore_spin_button():
            self.spin_button.config(
                text="⟳ SPIN",
                bg="#1976d2",
                activebackground="#1565c0"
            )

        if self.busy:
            return

        self.busy = True
        self.set_busy_buttons(True)

        self.spin_button.config(
            text="⟳ RUN...",
            bg="#f59e0b",
            activebackground="#d97706"
        )

        self.status_label.config(
            text="Bắt đầu chuỗi chuyển động..."
        )

        def worker():
            try:
                task()

                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text=(
                            "Đã hoàn thành: "
                            "tiến → lùi → quay trái → quay phải."
                        )
                    )
                )

            except Exception as e:
                error_message = str(e)

                self.root.after(
                    0,
                    lambda message=error_message: messagebox.showerror(
                        "Movement Error",
                        message
                    )
                )

                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text="Chuỗi chuyển động thất bại."
                    )
                )

            finally:
                self.busy = False

                self.root.after(0, restore_spin_button)
                self.root.after(
                    0,
                    lambda: self.set_busy_buttons(False)
                )
                self.root.after(
                    0,
                    self.refresh_status_once
                )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()
        

    def on_robot(self):
        """
        Nếu robot hoặc lidar đang chạy/failed/partial -> stop robot + lidar + rviz.
        Nếu cả hai đang inactive -> start robot Pi3 + lidar Pi4 + rviz default.
        """
        def task():
            robot_state = remote_pi3_is_active(PI3_ROBOT_SERVICE)
            lidar_state = local_is_active(PI4_LIDAR_SERVICE)
            group = self.check_group_state([robot_state, lidar_state])

            if group == "inactive":
                rc, out, err = remote_pi3_systemctl("start", PI3_ROBOT_SERVICE)
                if rc != 0:
                    raise RuntimeError(f"Start robot on Pi3 failed:\n{err or out}")

                rc, out, err = local_systemctl("start", PI4_LIDAR_SERVICE)
                if rc != 0:
                    raise RuntimeError(f"Start lidar on Pi4 failed:\n{err or out}")

                start_rviz_with_file(RVIZ_DEFAULT)
            else:
                stop_current_rviz()
                local_systemctl("stop", PI4_LIDAR_SERVICE)
                remote_pi3_systemctl("stop", PI3_ROBOT_SERVICE)

        self.run_background(task, "Robot toggled")

    def on_slam(self):
        """
        Nếu slam đang inactive -> start slam, đổi RViz sang slam.rviz.
        Nếu slam đang active/failed -> stop slam, stop RViz.
        """
        def task():
            state = local_is_active(PI4_SLAM_SERVICE)

            if state == "inactive":
                rc, out, err = local_systemctl("start", PI4_SLAM_SERVICE)
                if rc != 0:
                    raise RuntimeError(f"Start slam failed:\n{err or out}")

                start_rviz_with_file(RVIZ_SLAM)
            else:
                stop_current_rviz()
                local_systemctl("stop", PI4_SLAM_SERVICE)

        self.run_background(task, "SLAM toggled")

    def on_savemap(self):
        """
        Save map là oneshot. Bấm là gọi một phát.
        Không toggle như service chạy nền.
        """
        def task():
            map_name = clean_map_name(self.map_entry.get())
            service = f"savemap@{map_name}.service"

            rc, out, err = local_systemctl("start", service)
            if rc != 0:
                raise RuntimeError(f"Save map failed:\n{err or out}")

        self.run_background(task, "Save map command sent")

    def on_loadmap(self):
        """
        Toggle loadmap@<map_name>.service.

        Khi bật:
        - start map + AMCL;
        - nếu có <map_name>_keepout.yaml thì start Keepout Filter;
        - đặt initial pose AMCL tại x=0, y=0, yaw=0.

        Khi tắt:
        - stop Keepout Filter;
        - stop map + AMCL.
        """

        def task():
            map_name = clean_map_name(self.map_entry.get())

            loadmap_service = f"loadmap@{map_name}.service"
            keepout_service = f"keepout@{map_name}.service"

            keepout_yaml = (
                f"/home/pi/maps/"
                f"{map_name}_keepout.yaml"
            )

            state = local_is_active(loadmap_service)

            if state == "inactive":
                # ==========================================
                # 1. Khởi động map và AMCL
                # ==========================================
                rc, out, err = local_systemctl(
                    "start",
                    loadmap_service
                )

                if rc != 0:
                    raise RuntimeError(
                        f"Start loadmap failed:\n{err or out}"
                    )

                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text="Đang khởi động map và AMCL..."
                    )
                )

                time.sleep(3.0)

                # ==========================================
                # 2. Kiểm tra và khởi động Keepout
                # ==========================================
                if os.path.isfile(keepout_yaml):
                    rc, out, err = local_systemctl(
                        "start",
                        keepout_service
                    )

                    if rc != 0:
                        raise RuntimeError(
                            "Map đã load nhưng Keepout Filter "
                            f"khởi động thất bại:\n{err or out}"
                        )

                    keepout_status = "Có vùng cấm"
                else:
                    # Bảo đảm instance cũ không còn chạy
                    local_systemctl(
                        "stop",
                        keepout_service
                    )

                    keepout_status = "Không có vùng cấm"

                # ==========================================
                # 3. Đặt initial pose AMCL: 0, 0, 0
                # ==========================================
                initial_pose = (
                    "{"
                    "header: {"
                    "stamp: now, "
                    "frame_id: 'map'"
                    "}, "
                    "pose: {"
                    "pose: {"
                    "position: {"
                    "x: 0.0, y: 0.0, z: 0.0"
                    "}, "
                    "orientation: {"
                    "x: 0.0, y: 0.0, z: 0.0, w: 1.0"
                    "}"
                    "}, "
                    "covariance: ["
                    "0.25, 0.0, 0.0, 0.0, 0.0, 0.0, "
                    "0.0, 0.25, 0.0, 0.0, 0.0, 0.0, "
                    "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
                    "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
                    "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
                    "0.0, 0.0, 0.0, 0.0, 0.0, 0.0685"
                    "]"
                    "}"
                    "}"
                )

                cmd = (
                    "ros2 topic pub --once "
                    "/initialpose "
                    "geometry_msgs/msg/PoseWithCovarianceStamped "
                    f"{shlex.quote(initial_pose)}"
                )

                rc, out, err = run_cmd(
                    cmd,
                    timeout=15
                )

                if rc != 0:
                    raise RuntimeError(
                        "Không đặt được initial pose AMCL:\n"
                        f"{err or out}"
                    )

                self.root.after(
                    0,
                    lambda status=keepout_status:
                    self.status_label.config(
                        text=(
                            f"Đã load '{map_name}'. "
                            f"{status}. "
                            "AMCL = (0, 0, 0°)"
                        )
                    )
                )

            else:
                # ==========================================
                # Tắt Keepout trước
                # ==========================================
                local_systemctl(
                    "stop",
                    keepout_service
                )

                # Tắt map và AMCL
                rc, out, err = local_systemctl(
                    "stop",
                    loadmap_service
                )

                if rc != 0:
                    raise RuntimeError(
                        f"Stop loadmap failed:\n{err or out}"
                    )

        self.run_background(task)

    def on_nav(self):
        """
        Nếu nav inactive -> start nav, đổi RViz sang nav.rviz.
        Nếu nav active/failed -> stop nav, stop RViz.
        """
        def task():
            state = local_is_active(PI4_NAV_SERVICE)

            if state == "inactive":
                
                start_rviz_with_file(RVIZ_NAV)
                
                rc, out, err = local_systemctl("start", PI4_NAV_SERVICE)
                if rc != 0:
                    raise RuntimeError(f"Start nav failed:\n{err or out}")

                
            else:
                stop_current_rviz()
                local_systemctl("stop", PI4_NAV_SERVICE)

        self.run_background(task, "Nav toggled")

    def on_tomato(self):
        """
        Toggle tomato.service.
        """
        def task():
            state = local_is_active(PI4_TOMATO_SERVICE)

            if state == "inactive":
                rc, out, err = local_systemctl("start", PI4_TOMATO_SERVICE)
                if rc != 0:
                    raise RuntimeError(f"Start tomato failed:\n{err or out}")
            else:
                local_systemctl("stop", PI4_TOMATO_SERVICE)

        self.run_background(task, "Tomato toggled")


if __name__ == "__main__":
    rclpy.init()

    root = tk.Tk()
    app = RobotControlGUI(root)
    root.mainloop()