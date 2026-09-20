#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from PIL import Image
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


OUTPUT_PATH = Path("keepout_filter_mask.png")


class KeepoutMaskSaver(Node):
    def __init__(self) -> None:
        super().__init__("keepout_mask_saver")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.subscription = self.create_subscription(
            OccupancyGrid,
            "/keepout_filter_mask",
            self.mask_callback,
            qos,
        )

        self.received = False
        self.get_logger().info(
            "Đang chờ dữ liệu từ /keepout_filter_mask..."
        )

    def mask_callback(self, msg: OccupancyGrid) -> None:
        if self.received:
            return

        self.received = True

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin.position

        data = np.asarray(msg.data, dtype=np.int16)

        expected_size = width * height
        if data.size != expected_size:
            self.get_logger().error(
                f"Kích thước dữ liệu không hợp lệ: "
                f"{data.size}, mong đợi {expected_size}"
            )
            rclpy.shutdown()
            return

        # OccupancyGrid lưu dữ liệu theo hàng:
        # index = x + y * width
        grid = data.reshape((height, width))

        # ROS dùng gốc tọa độ ở góc dưới trái,
        # ảnh PNG dùng gốc ở góc trên trái.
        grid = np.flipud(grid)

        # Ảnh RGB để dễ phân biệt:
        # - Free 0       : trắng
        # - Keepout 100  : đen
        # - Unknown -1   : xám
        # - Trung gian   : thang xám
        image_array = np.zeros((height, width, 3), dtype=np.uint8)

        unknown_mask = grid < 0
        known_mask = ~unknown_mask

        # 0 -> 255, 100 -> 0
        gray = np.clip(
            255.0 * (1.0 - grid.astype(np.float32) / 100.0),
            0,
            255,
        ).astype(np.uint8)

        image_array[known_mask, 0] = gray[known_mask]
        image_array[known_mask, 1] = gray[known_mask]
        image_array[known_mask, 2] = gray[known_mask]

        image_array[unknown_mask] = [128, 128, 128]

        image = Image.fromarray(image_array, mode="RGB")
        image.save(OUTPUT_PATH)

        unique_values, counts = np.unique(data, return_counts=True)

        self.get_logger().info(f"Đã lưu ảnh: {OUTPUT_PATH.resolve()}")
        self.get_logger().info(
            f"Kích thước: {width} x {height} pixel"
        )
        self.get_logger().info(
            f"Resolution: {resolution:.6f} m/pixel"
        )
        self.get_logger().info(
            f"Origin: x={origin.x:.3f}, "
            f"y={origin.y:.3f}, z={origin.z:.3f}"
        )
        self.get_logger().info(
            f"Frame: {msg.header.frame_id!r}"
        )

        print("\nPhân bố giá trị trong mask:")
        for value, count in zip(unique_values, counts):
            meaning = {
                -1: "unknown",
                0: "free",
                100: "keepout",
            }.get(int(value), "intermediate")

            print(
                f"  value={int(value):4d}: "
                f"{int(count):8d} pixels ({meaning})"
            )

        rclpy.shutdown()


def main() -> None:
    rclpy.init()
    node = KeepoutMaskSaver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()