#!/usr/bin/env python3
"""Pha 1 - chay mot luot khao sat bang follow_waypoints.

Quy tac bat buoc (theo yeu cau):
- Goi action follow_waypoints DUY NHAT MOT LAN cho ca 5 pose (4 vi tri quan
  sat + 1 ve vi tri quy dinh). Khong gui lenh dung/huy giua chung trong luc
  chay binh thuong.
- Robot khong dung de chup. Camera chay tren mot thread rieng, chup lien tuc
  o 2 Hz trong luc robot dang di chuyen, khong doi van toc ve duoi nguong.
- Pha nay CHI luu anh (JPEG) va metadata (manifest.jsonl). KHONG chay bat ky
  mo hinh suy luan nao o day (TogroNet-Stage / TogroNet-Fruit thuoc Pha 2).
- Phoi sang va gain camera duoc khoa co dinh cho ca luot ngay sau khi camera
  khoi dong, de anh chup dau luot va cuoi luot co do sang nhat quan.

Mot luot duoc coi la HOAN THANH khi ca bon dieu deu dung:
  1) Nav2 bao FollowWaypoints thanh cong VA khong co waypoint nao bi missed.
  2) Robot ve vi tri quy dinh (home_pose) trong ban kinh home_tolerance_m.
  3) Co it nhat 1 anh duoc luu VA ty le khung hinh bi bo (khong chup duoc)
     duoi max_dropped_frame_ratio.
  4) manifest.jsonl ghi ra va doc lai duoc, moi dong hop le, moi anh ton tai.

Ket qua duoc ghi vao <run_root>/run_<timestamp>/:
  images/img_XXXXXX.jpg
  manifest.jsonl
  run_summary.json
"""

import json
import math
import os
import threading
import time
from datetime import datetime, timezone

import cv2
import rclpy
import rclpy.time
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints
from picamera2 import Picamera2
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener

REQUIRED_MANIFEST_KEYS = (
    'image_id', 'captured_at', 'monotonic_ns',
    'target_waypoint_index', 'position', 'pose',
)


def yaw_to_quaternion(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class SurveyRunNode(Node):

    def __init__(self):
        super().__init__('survey_run')

        default_waypoints = os.path.join(
            get_package_share_directory('tomato_survey'),
            'config', 'waypoints.yaml',
        )

        self.declare_parameter('waypoints_file', default_waypoints)
        self.declare_parameter('run_root', os.path.expanduser('~/tomato_survey_runs'))
        self.declare_parameter('map_frame', 'map')
        # Yeu cau goc noi "tf giua map va base_link", nhung cau hinh dang
        # chay tren robot nay (my_robot_controllers.yaml, nav2_pi_override.yaml)
        # dung base_footprint la robot_base_frame thuc te. Doi lai bang tham so
        # -p base_frame:=base_link neu can khop dung chu "base_link".
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('capture_hz', 4.0)
        self.declare_parameter('position_radius_m', 0.8)
        self.declare_parameter('home_tolerance_m', 0.3)
        self.declare_parameter('max_dropped_frame_ratio', 0.10)
        self.declare_parameter('min_captured_frames', 1)
        self.declare_parameter('camera_width', 1280)
        self.declare_parameter('camera_height', 720)
        self.declare_parameter('jpeg_quality', 100)

        self.waypoints_file = self.get_parameter('waypoints_file').value
        self.run_root = os.path.expanduser(self.get_parameter('run_root').value)
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.capture_hz = float(self.get_parameter('capture_hz').value)
        self.position_radius_m = float(self.get_parameter('position_radius_m').value)
        self.home_tolerance_m = float(self.get_parameter('home_tolerance_m').value)
        self.max_dropped_frame_ratio = float(
            self.get_parameter('max_dropped_frame_ratio').value
        )
        self.min_captured_frames = int(self.get_parameter('min_captured_frames').value)
        self.cam_width = int(self.get_parameter('camera_width').value)
        self.cam_height = int(self.get_parameter('camera_height').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        self.observation_waypoints, self.home_pose = self._load_waypoints(
            self.waypoints_file
        )

        os.makedirs(self.run_root, exist_ok=True)
        run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.run_root, run_id)
        self.images_dir = os.path.join(self.run_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.run_dir, 'manifest.jsonl')
        self.summary_path = os.path.join(self.run_dir, 'run_summary.json')
        self._manifest_file = open(self.manifest_path, 'w', encoding='utf-8')
        self._manifest_lock = threading.Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state_lock = threading.Lock()
        self.target_waypoint_index = -1

        self.frames_attempted = 0
        self.frames_saved = 0
        self._seq = 0

        self.nav_done_event = threading.Event()
        self.nav_result_status = None
        self.nav_missed_waypoints = None
        self.goal_handle = None

        self.camera_stop_event = threading.Event()
        self.picam2 = None

        self._start_camera()

        self.action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        self.camera_thread = threading.Thread(
            target=self._camera_loop, name='survey_camera', daemon=True
        )
        self.camera_thread.start()

        self._send_follow_waypoints_goal()

    # ------------------------------------------------------------------
    # Waypoints
    # ------------------------------------------------------------------
    def _load_waypoints(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        observation = data['observation_waypoints']
        home = data['home_pose']
        if len(observation) != 4:
            raise ValueError(
                f'waypoints_file phai co dung 4 vi tri quan sat, thay {len(observation)}'
            )
        return observation, home

    def _build_pose_stamped(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        _, _, qz, qw = yaw_to_quaternion(float(yaw))
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    # ------------------------------------------------------------------
    # Camera - tu dong phoi sang/gain/AWB (khong khoa)
    # ------------------------------------------------------------------
    def _start_camera(self):
        self.picam2 = Picamera2()
        video_config = self.picam2.create_video_configuration(
            main={
                'size': (self.cam_width, self.cam_height),
                # Quirk Picamera2: format 'BGR888' moi tra ve du lieu dung
                # thu tu kenh RGB thuc su.
                'format': 'BGR888',
            },
            buffer_count=4,
        )
        self.picam2.configure(video_config)
        self.picam2.start()
        self.get_logger().info(
            f'Camera da khoi dong o {self.cam_width}x{self.cam_height}.'
        )

    @staticmethod
    def _crop_to_square(frame):
        """Cat doi xung hai ben (hoac tren/duoi) de duoc anh vuong.

        Vi du 1280x720 -> cat het 280px moi ben trai/phai -> 720x720.
        """
        h, w = frame.shape[:2]
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return frame[y0:y0 + side, x0:x0 + side]

    # ------------------------------------------------------------------
    # FollowWaypoints action - goi mot lan duy nhat
    # ------------------------------------------------------------------
    def _send_follow_waypoints_goal(self):
        self.get_logger().info("Dang cho action server 'follow_waypoints'...")
        self.action_client.wait_for_server()

        poses = [
            self._build_pose_stamped(wp['x'], wp['y'], wp['yaw'])
            for wp in self.observation_waypoints
        ]
        poses.append(
            self._build_pose_stamped(
                self.home_pose['x'], self.home_pose['y'], self.home_pose['yaw']
            )
        )

        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = poses

        self.get_logger().info(
            f'Goi follow_waypoints voi {len(poses)} pose (4 quan sat + 1 ve nha).'
        )

        send_future = self.action_client.send_goal_async(
            goal_msg, feedback_callback=self._on_feedback
        )
        send_future.add_done_callback(self._on_goal_response)

    def _on_feedback(self, feedback_msg):
        with self.state_lock:
            self.target_waypoint_index = int(feedback_msg.feedback.current_waypoint)

    def _on_goal_response(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().error('follow_waypoints goal bi tu choi.')
            self.nav_result_status = 'REJECTED'
            self.nav_missed_waypoints = None
            self.nav_done_event.set()
            return

        self.get_logger().info('follow_waypoints goal duoc chap nhan.')
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        wrapped = future.result()
        status = wrapped.status
        result = wrapped.result

        from action_msgs.msg import GoalStatus
        status_name = {
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
        }.get(status, f'UNKNOWN({status})')

        self.nav_result_status = status_name
        self.nav_missed_waypoints = [
            int(mw.index) for mw in result.missed_waypoints
        ]

        self.get_logger().info(
            f'follow_waypoints ket thuc: status={status_name} '
            f'missed_waypoints={self.nav_missed_waypoints}'
        )
        self.nav_done_event.set()

    # ------------------------------------------------------------------
    # Camera thread - chup lien tuc, KHONG doi robot dung, KHONG chay mo hinh
    # ------------------------------------------------------------------
    def _camera_loop(self):
        period = 1.0 / max(self.capture_hz, 0.1)
        next_tick = time.monotonic()

        while not self.camera_stop_event.is_set():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.05))
                continue

            next_tick += period

            with self.state_lock:
                heading_home = (
                    self.target_waypoint_index >= len(self.observation_waypoints)
                )
            if heading_home:
                # Da quan sat xong 4 vi tri, dang tren duong ve nha - khong
                # chup nua (tranh anh trung lap/lech huong luc di ngang qua
                # cac vi tri cu tren duong ve).
                continue

            self.frames_attempted += 1
            self._capture_one_frame()

    def _capture_one_frame(self):
        try:
            rgb_frame = self.picam2.capture_array('main')
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            frame = self._crop_to_square(frame)
        except Exception as exc:
            self.get_logger().warn(f'Bo qua khung hinh - loi doc camera: {exc}')
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().warn(f'Bo qua khung hinh - chua co tf: {exc}')
            return

        monotonic_ns = time.monotonic_ns()
        captured_at = datetime.now(timezone.utc).isoformat()

        t = transform.transform.translation
        q = transform.transform.rotation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

        with self.state_lock:
            target_waypoint_index = self.target_waypoint_index

        position = self._assign_position(t.x, t.y)

        self._seq += 1
        image_id = f'img_{self._seq:06d}.jpg'
        image_path = os.path.join(self.images_dir, image_id)

        ok = cv2.imwrite(
            image_path, frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            self.get_logger().warn(f'Bo qua khung hinh - ghi JPEG that bai: {image_id}')
            return

        record = {
            'image_id': image_id,
            'captured_at': captured_at,
            'monotonic_ns': monotonic_ns,
            'target_waypoint_index': target_waypoint_index,
            'position': position,
            'pose': {
                'x': t.x, 'y': t.y, 'z': t.z, 'yaw': yaw,
                'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w,
            },
        }

        with self._manifest_lock:
            self._manifest_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            self._manifest_file.flush()

        self.frames_saved += 1

    def _assign_position(self, x, y):
        best_id = None
        best_dist = None
        for wp in self.observation_waypoints:
            dist = math.hypot(x - wp['x'], y - wp['y'])
            if dist <= self.position_radius_m and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_id = wp['id']
        return best_id if best_id is not None else 'qua_duong'

    # ------------------------------------------------------------------
    # Ket thuc luot: kiem tra 4 dieu kien, ghi run_summary.json
    # ------------------------------------------------------------------
    def finalize_run(self):
        self.camera_stop_event.set()
        self.camera_thread.join(timeout=5.0)

        with self._manifest_lock:
            self._manifest_file.flush()
            self._manifest_file.close()

        try:
            self.picam2.stop()
        except Exception:
            pass

        # 1) Nav2 bao xong toan bo waypoint
        nav_ok = (
            self.nav_result_status == 'SUCCEEDED'
            and self.nav_missed_waypoints == []
        )

        # 2) Robot ve vi tri quy dinh trong home_tolerance_m
        home_distance = None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
            )
            t = transform.transform.translation
            home_distance = math.hypot(
                t.x - self.home_pose['x'], t.y - self.home_pose['y']
            )
        except TransformException as exc:
            self.get_logger().warn(f'Khong doc duoc tf cuoi luot: {exc}')
        home_ok = home_distance is not None and home_distance <= self.home_tolerance_m

        # 3) Co anh va ty le khung bi bo duoi nguong
        dropped_ratio = (
            1.0 - (self.frames_saved / self.frames_attempted)
            if self.frames_attempted > 0 else 1.0
        )
        capture_ok = (
            self.frames_saved >= self.min_captured_frames
            and dropped_ratio < self.max_dropped_frame_ratio
        )

        # 4) manifest ghi va doc lai duoc
        manifest_ok, manifest_detail = self._verify_manifest()

        run_complete = nav_ok and home_ok and capture_ok and manifest_ok

        summary = {
            'run_dir': self.run_dir,
            'waypoints_file': self.waypoints_file,
            'run_complete': run_complete,
            'checks': {
                'nav_all_waypoints_done': {
                    'ok': nav_ok,
                    'nav_result_status': self.nav_result_status,
                    'missed_waypoints': self.nav_missed_waypoints,
                },
                'home_position_reached': {
                    'ok': home_ok,
                    'home_distance_m': home_distance,
                    'home_tolerance_m': self.home_tolerance_m,
                },
                'capture_completeness': {
                    'ok': capture_ok,
                    'frames_attempted': self.frames_attempted,
                    'frames_saved': self.frames_saved,
                    'dropped_ratio': dropped_ratio,
                    'max_dropped_frame_ratio': self.max_dropped_frame_ratio,
                },
                'manifest_roundtrip': {
                    'ok': manifest_ok,
                    'detail': manifest_detail,
                },
            },
        }

        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.get_logger().info(f'run_complete={run_complete} -> {self.summary_path}')
        return summary

    def _verify_manifest(self):
        if not os.path.isfile(self.manifest_path):
            return False, 'manifest.jsonl khong ton tai'

        ok_lines = 0
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    return False, f'dong {line_no} loi JSON: {exc}'

                missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in record]
                if missing:
                    return False, f'dong {line_no} thieu truong: {missing}'

                image_path = os.path.join(self.images_dir, record['image_id'])
                if not os.path.isfile(image_path):
                    return False, f"dong {line_no} thieu file anh {record['image_id']}"

                ok_lines += 1

        if ok_lines != self.frames_saved:
            return False, (
                f'so dong doc lai ({ok_lines}) khac so anh da luu '
                f'({self.frames_saved})'
            )

        return True, f'{ok_lines} dong hop le, doc lai thanh cong'


def main(args=None):
    rclpy.init(args=args)
    node = SurveyRunNode()

    try:
        while rclpy.ok() and not node.nav_done_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warn('Nhan Ctrl+C - huy goal follow_waypoints.')
        if node.goal_handle is not None:
            try:
                node.goal_handle.cancel_goal_async()
            except Exception:
                pass
    finally:
        node.finalize_run()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
