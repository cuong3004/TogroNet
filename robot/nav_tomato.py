#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot di chuyển liên tục giữa hai điểm A và B.

Quy tắc:
- Đang đi A -> B:
    + Thành công: đổi mục tiêu về A.
    + Thất bại: cũng đổi mục tiêu về A.
- Đang đi B -> A:
    + Thành công: đổi mục tiêu sang B.
    + Thất bại: cũng đổi mục tiêu sang B.

Do đó thứ tự mục tiêu luôn là:
    B -> A -> B -> A -> ...

Giả thiết ban đầu robot đang ở gần điểm A.
"""

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

FRAME_ID = "map"

# Nhập tọa độ hai điểm tại đây.
POINT_A = (0.00, 0.00)
POINT_B = (4.00, 0.39)

# Robot bắt đầu bằng việc đi từ A tới B.
FIRST_TARGET = "B"

WAIT_AFTER_SUCCESS_SEC = 0.5
WAIT_AFTER_FAILURE_SEC = 2.0
GOAL_TIMEOUT_SEC = 300.0
CLEAR_LOCAL_COSTMAP_AFTER_FAILURE = True


def create_pose(navigator: BasicNavigator, x: float, y: float, yaw: float) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = FRAME_ID
    pose.header.stamp = navigator.get_clock().now().to_msg()

    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = 0.0

    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose


def create_two_points(navigator: BasicNavigator):
    ax, ay = POINT_A
    bx, by = POINT_B

    yaw_a_to_b = math.atan2(by - ay, bx - ax)
    yaw_b_to_a = math.atan2(ay - by, ax - bx)

    return {
        "A": create_pose(navigator, ax, ay, yaw_b_to_a),
        "B": create_pose(navigator, bx, by, yaw_a_to_b),
    }


def task_is_complete(navigator: BasicNavigator, task) -> bool:
    try:
        return navigator.isTaskComplete(task=task)
    except TypeError:
        return navigator.isTaskComplete()


def get_task_result(navigator: BasicNavigator, task):
    try:
        return navigator.getResult(task=task)
    except TypeError:
        return navigator.getResult()


def cancel_task(navigator: BasicNavigator, task) -> None:
    try:
        navigator.cancelTask(task=task)
    except TypeError:
        navigator.cancelTask()


def navigate_once(
    navigator: BasicNavigator,
    target_name: str,
    target_pose: PoseStamped,
) -> bool:
    """Gửi một goal duy nhất, không retry lại cùng điểm."""
    target_pose.header.stamp = navigator.get_clock().now().to_msg()

    navigator.info(
        f"Đi tới {target_name}: "
        f"x={target_pose.pose.position.x:.3f}, "
        f"y={target_pose.pose.position.y:.3f}"
    )

    task = navigator.goToPose(target_pose)

    if task is None:
        navigator.error(f"Không gửi được goal tới {target_name}.")
        return False

    start_time = time.monotonic()

    while rclpy.ok() and not task_is_complete(navigator, task):
        if time.monotonic() - start_time > GOAL_TIMEOUT_SEC:
            navigator.error(f"Đi tới {target_name} bị timeout, hủy goal.")
            cancel_task(navigator, task)
            return False

        time.sleep(0.1)

    if not rclpy.ok():
        return False

    result = get_task_result(navigator, task)

    if result == TaskResult.SUCCEEDED:
        navigator.info(f"Đã tới {target_name}.")
        return True

    if result == TaskResult.CANCELED:
        navigator.warn(f"Goal tới {target_name} bị hủy.")
    elif result == TaskResult.FAILED:
        navigator.error(f"Đi tới {target_name} thất bại.")
    else:
        navigator.error(f"Kết quả không xác định: {result}")

    return False


def main() -> None:
    rclpy.init()
    navigator = BasicNavigator(node_name="two_point_loop_navigator")

    try:
        navigator.info("Đang chờ Nav2 active...")
        navigator.waitUntilNav2Active()
        navigator.info("Nav2 đã active.")

        poses = create_two_points(navigator)
        target_name = FIRST_TARGET

        while rclpy.ok():
            succeeded = navigate_once(
                navigator=navigator,
                target_name=target_name,
                target_pose=poses[target_name],
            )

            if succeeded:
                time.sleep(WAIT_AFTER_SUCCESS_SEC)
            else:
                if CLEAR_LOCAL_COSTMAP_AFTER_FAILURE:
                    try:
                        navigator.info("Lỗi, xóa local costmap...")
                        navigator.clearLocalCostmap()
                    except Exception as error:
                        navigator.warn(f"Không xóa được local costmap: {error}")

                time.sleep(WAIT_AFTER_FAILURE_SEC)

            # Thành công hay thất bại đều đổi sang điểm còn lại.
            target_name = "A" if target_name == "B" else "B"

    except KeyboardInterrupt:
        navigator.warn("Nhận Ctrl+C, đang dừng.")

    except Exception as error:
        navigator.error(f"Lỗi chương trình: {error}")

    finally:
        try:
            navigator.cancelTask()
        except Exception:
            pass

        navigator.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
