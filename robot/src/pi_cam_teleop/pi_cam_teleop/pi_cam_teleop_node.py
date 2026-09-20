#!/usr/bin/env python3
"""Teleop ban phim + stream Pi Camera + chup anh bang phim 'c'.

Chay giong teleop_twist_keyboard nhung dieu khien phim duoc doc truc tiep
tu cua so hien thi camera (OpenCV) thay vi tu terminal, vi vay cua so
camera phai dang duoc focus khi go phim.

Vi du chay tuong duong lenh teleop_twist_keyboard ban dang dung:

    ros2 run pi_cam_teleop pi_cam_teleop \\
        --ros-args \\
        -r cmd_vel:=/diff_drive_controller/cmd_vel \\
        -p stamped:=true
"""

import json
import os
import threading
import time
from datetime import datetime

import cv2
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from picamera2 import Picamera2
from rclpy.node import Node

MOVE_BINDINGS = {
    'i': (1, 0),
    ',': (-1, 0),
    'j': (0, 1),
    'l': (0, -1),
    'u': (1, 1),
    'o': (1, -1),
    'm': (-1, -1),
    '.': (-1, 1),
}

# (linear_scale, angular_scale)
SPEED_BINDINGS = {
    'q': (1.1, 1.1),
    'z': (0.9, 0.9),
    'w': (1.1, 1.0),
    'x': (0.9, 1.0),
    'e': (1.0, 1.1),
    'r': (1.0, 0.9),
}

CAPTURE_KEY = ord('c')
AWB_AUTO_KEY = ord('a')
WB_SAVE_KEY = ord('s')
QUIT_KEYS = (27, ord('Q'))  # ESC

HELP_MSG = """
Dieu khien robot (giong teleop_twist_keyboard) - go phim TREN CUA SO CAMERA:
---------------------------
   u    i    o
   j    k    l
   m    ,    .

i / , : tien / lui
j / l  : quay trai / phai tai cho
u / o  : tien + re trai / phai
m / .  : lui + re trai / phai
phim khac (vd k) : dung robot

q/z : tang/giam ca toc do dai va toc do quay
w/x : tang/giam toc do dai
e/r : tang/giam toc do quay

c   : CHUP ANH tu camera
a   : bat CHE DO TU DONG can bang trang (AWB auto)
s   : LUU & KHOA he so can bang trang hien tai (dung file cu tu day)
ESC : thoat chuong trinh
---------------------------
"""


class PiCamTeleopNode(Node):

    def __init__(self):
        super().__init__('pi_cam_teleop')

        self.declare_parameter('stamped', False)
        self.declare_parameter('frame_id', '')
        self.declare_parameter('speed', 0.2)
        self.declare_parameter('turn', 1.0)
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('camera_fps', 30.0)
        self.declare_parameter(
            'save_dir', os.path.expanduser('~/Pictures/pi_cam_teleop')
        )
        self.declare_parameter('window_name', 'Pi Cam Teleop')
        self.declare_parameter(
            'wb_gains_file',
            os.path.expanduser('~/.pi_cam_teleop/wb_gains.json'),
        )

        self.stamped = self.get_parameter('stamped').value
        self.frame_id = self.get_parameter('frame_id').value
        self.speed = float(self.get_parameter('speed').value)
        self.turn = float(self.get_parameter('turn').value)
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.cam_width = int(self.get_parameter('camera_width').value)
        self.cam_height = int(self.get_parameter('camera_height').value)
        self.cam_fps = float(self.get_parameter('camera_fps').value)
        self.save_dir = os.path.expanduser(
            self.get_parameter('save_dir').value
        )
        self.window_name = self.get_parameter('window_name').value
        self.wb_gains_file = os.path.expanduser(
            self.get_parameter('wb_gains_file').value
        )

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.wb_gains_file), exist_ok=True)

        msg_type = TwistStamped if self.stamped else Twist
        self.cmd_vel_pub = self.create_publisher(msg_type, 'cmd_vel', 10)

        self.linear_dir = 0.0
        self.angular_dir = 0.0

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.picam2 = None
        self.camera_stop_event = threading.Event()

        self._start_camera()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info(HELP_MSG)
        self.get_logger().info(
            f'cmd_vel type: {"TwistStamped" if self.stamped else "Twist"} | '
            f'speed={self.speed:.2f} m/s | turn={self.turn:.2f} rad/s'
        )
        self.get_logger().info(f'Anh chup se duoc luu vao: {self.save_dir}')
        self.get_logger().info(f'File he so can bang trang: {self.wb_gains_file}')

        timer_period = 1.0 / max(publish_rate_hz, 1.0)
        self.timer = self.create_timer(timer_period, self._on_timer)

    def _start_camera(self):
        try:
            self.picam2 = Picamera2()
            video_config = self.picam2.create_video_configuration(
                main={
                    'size': (self.cam_width, self.cam_height),
                    # Luu y: quirk cua Picamera2 - format 'BGR888' moi tra ve
                    # du lieu theo thu tu kenh RGB thuc su (va nguoc lai).
                    'format': 'BGR888',
                },
                controls={'FrameRate': self.cam_fps},
                buffer_count=4,
            )
            self.picam2.configure(video_config)
            self.picam2.start()
            self._load_saved_white_balance()

            self.camera_thread = threading.Thread(
                target=self._camera_loop,
                name='picamera2_capture',
                daemon=True,
            )
            self.camera_thread.start()
        except Exception as exc:
            self.get_logger().error(f'Khong the khoi dong Pi Camera: {exc}')

    def _camera_loop(self):
        time.sleep(1.0)
        while not self.camera_stop_event.is_set():
            try:
                rgb_frame = self.picam2.capture_array('main')
                if rgb_frame is None or rgb_frame.size == 0:
                    continue
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                with self.frame_lock:
                    self.latest_frame = frame
            except Exception as exc:
                self.get_logger().error(f'Loi doc camera: {exc}')
                time.sleep(0.5)

    def _on_timer(self):
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()

        if frame is not None:
            display = frame.copy()
            overlay = (
                f'speed={self.speed:.2f} turn={self.turn:.2f} '
                f"| lin={self.linear_dir:.0f} ang={self.angular_dir:.0f} | 'c'=chup anh"
            )
            cv2.putText(
                display, overlay, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.imshow(self.window_name, display)

        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key, frame)
        self._publish_cmd_vel()

    def _handle_key(self, key, frame):
        if key in (255, 0xFF):
            return  # khong co phim nao duoc nhan trong khung nay

        char = chr(key) if key < 128 else ''

        if key in QUIT_KEYS:
            self.get_logger().info('Nhan ESC/Q - dang thoat...')
            rclpy.shutdown()
            return

        if key == CAPTURE_KEY:
            self._capture_photo(frame)
            return

        if key == AWB_AUTO_KEY:
            self._set_auto_white_balance()
            return

        if key == WB_SAVE_KEY:
            self._save_and_lock_white_balance()
            return

        if char in MOVE_BINDINGS:
            self.linear_dir, self.angular_dir = MOVE_BINDINGS[char]
            return

        if char in SPEED_BINDINGS:
            lin_scale, ang_scale = SPEED_BINDINGS[char]
            self.speed *= lin_scale
            self.turn *= ang_scale
            self.get_logger().info(
                f'speed={self.speed:.3f} m/s  turn={self.turn:.3f} rad/s'
            )
            return

        # bat ky phim nao khac (vd 'k') -> dung robot
        self.linear_dir = 0.0
        self.angular_dir = 0.0

    def _capture_photo(self, frame):
        if frame is None:
            self.get_logger().warn('Chua co khung hinh nao tu camera de chup.')
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filepath = os.path.join(self.save_dir, f'capture_{timestamp}.jpg')
        if cv2.imwrite(filepath, frame):
            self.get_logger().info(f'Da chup anh: {filepath}')
        else:
            self.get_logger().error(f'Chup anh that bai: {filepath}')

    def _load_saved_white_balance(self):
        if not os.path.isfile(self.wb_gains_file):
            return
        try:
            with open(self.wb_gains_file) as f:
                data = json.load(f)
            red_gain = float(data['red_gain'])
            blue_gain = float(data['blue_gain'])
            self.picam2.set_controls({
                'AwbEnable': False,
                'ColourGains': (red_gain, blue_gain),
            })
            self.get_logger().info(
                'Da nap he so can bang trang da luu: '
                f'red={red_gain:.3f} blue={blue_gain:.3f}'
            )
        except Exception as exc:
            self.get_logger().error(f'Khong doc duoc file can bang trang: {exc}')

    def _set_auto_white_balance(self):
        if self.picam2 is None:
            return
        try:
            self.picam2.set_controls({'AwbEnable': True})
            self.get_logger().info('Da bat che do tu dong can bang trang (AWB auto).')
        except Exception as exc:
            self.get_logger().error(f'Khong the bat AWB auto: {exc}')

    def _save_and_lock_white_balance(self):
        if self.picam2 is None:
            return
        try:
            metadata = self.picam2.capture_metadata()
            gains = metadata.get('ColourGains')
            if gains is None:
                self.get_logger().warn(
                    'Chua doc duoc he so can bang trang tu camera, thu lai sau.'
                )
                return
            red_gain, blue_gain = gains
            self.picam2.set_controls({
                'AwbEnable': False,
                'ColourGains': (red_gain, blue_gain),
            })
            with open(self.wb_gains_file, 'w') as f:
                json.dump(
                    {'red_gain': red_gain, 'blue_gain': blue_gain}, f
                )
            self.get_logger().info(
                'Da luu & khoa he so can bang trang: '
                f'red={red_gain:.3f} blue={blue_gain:.3f} -> {self.wb_gains_file}'
            )
        except Exception as exc:
            self.get_logger().error(f'Loi luu he so can bang trang: {exc}')

    def _publish_cmd_vel(self):
        linear_x = self.linear_dir * self.speed
        angular_z = self.angular_dir * self.turn

        if self.stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.twist.linear.x = linear_x
            msg.twist.angular.z = angular_z
        else:
            msg = Twist()
            msg.linear.x = linear_x
            msg.angular.z = angular_z

        self.cmd_vel_pub.publish(msg)

    def destroy_node(self):
        self.camera_stop_event.set()
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PiCamTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Gui lenh dung robot truoc khi thoat.
        node.linear_dir = 0.0
        node.angular_dir = 0.0
        try:
            node._publish_cmd_vel()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
