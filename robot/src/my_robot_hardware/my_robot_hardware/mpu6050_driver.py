#!/usr/bin/env python3
import math
import smbus
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

# =====================
# MPU6050 Registers
# =====================
DEVICE_ADDRESS = 0x68
PWR_MGMT_1     = 0x6B
GYRO_CONFIG    = 0x1B
GYRO_ZOUT_H    = 0x47

GYRO_FS_250 = 0x00  # ±250 deg/s

class MPU6050YawNode(Node):
    def __init__(self):
        super().__init__("mpu6050_yaw_node")

        # I2C
        self.bus = smbus.SMBus(1)
        self.init_mpu()

        # Publisher
        self.pub = self.create_publisher(
            Imu, "/imu/out", qos_profile_sensor_data
        )

        # Timer 100 Hz
        self.timer = self.create_timer(0.01, self.update)

        self.get_logger().info("MPU6050 yaw-only IMU node started")

    def init_mpu(self):
        self.bus.write_byte_data(DEVICE_ADDRESS, PWR_MGMT_1, 0x00)
        self.bus.write_byte_data(DEVICE_ADDRESS, GYRO_CONFIG, GYRO_FS_250)

    def read_int16(self, reg):
        high = self.bus.read_byte_data(DEVICE_ADDRESS, reg)
        low  = self.bus.read_byte_data(DEVICE_ADDRESS, reg + 1)
        value = (high << 8) | low
        if value > 32767:
            value -= 65536
        return value

    def update(self):
        try:
            # Read gyro Z raw
            gz_raw = self.read_int16(GYRO_ZOUT_H)

            # Convert to rad/s
            gz_deg = gz_raw * 250.0 / 32768.0
            gz_rad = gz_deg * math.pi / 180.0

            imu = Imu()
            imu.header.stamp = self.get_clock().now().to_msg()
            imu.header.frame_id = "base_footprint"

            # Only yaw rate
            imu.angular_velocity.z = gz_rad

            # Covariances
            imu.angular_velocity_covariance = [
                1e6, 0.0, 0.0,
                0.0, 1e6, 0.0,
                0.0, 0.0, 0.01   # trust yaw rate
            ]

            imu.linear_acceleration_covariance[0] = -1.0
            imu.orientation_covariance[0] = -1.0

            self.pub.publish(imu)

        except OSError:
            self.get_logger().warn("MPU6050 I2C read failed")

def main():
    rclpy.init()
    node = MPU6050YawNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()