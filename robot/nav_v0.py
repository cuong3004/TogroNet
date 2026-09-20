#!/usr/bin/env python3

import math
import rclpy

from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
#import tf_transformations


def create_pose_stamped(
    navigator: BasicNavigator,
    position_x: float,
    position_y: float,
    qz: float, qw: float
):
    """
    orientation_z:
        Góc yaw (radian)
    """

    # q_x, q_y, q_z, q_w = tf_transformations.quaternion_from_euler(
    #     0.0,
    #     0.0,
    #     orientation_z
    # )

    pose = PoseStamped()

    pose.header.frame_id = "map"
    pose.header.stamp = navigator.get_clock().now().to_msg()

    pose.pose.position.x = position_x
    pose.pose.position.y = position_y
    pose.pose.position.z = 0.0

    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw

    return pose


def main():
    print("Nav2 is start")
    rclpy.init()
    print("Nav2 is init")

    nav = BasicNavigator()

    # =========================================================
    # INITIAL POSE
    # =========================================================

    #initial_pose = create_pose_stamped(
    #    nav,
    #    0.0,   # x
    #    0.0,   # y
    #    0.0    # yaw
    #)

    #nav.setInitialPose(initial_pose)

    # =========================================================
    # WAIT NAV2 ACTIVE
    # =========================================================

    nav.waitUntilNav2Active()

    print("Nav2 is active")

    # =========================================================
    # DEFINE 2 GOALS
    # =========================================================

    point_a = create_pose_stamped(
        nav,
        0.0,           # x
        0.0,           # y
        0.0,
        1.0            # yaw
    )

    point_b = create_pose_stamped(
        nav,
        6.0,           # x
        0.0,           # y
        1.0,
        0.0        # yaw
    )

    # =========================================================
    # LOOP FOREVER
    # =========================================================

    current_target = "B"

    while rclpy.ok():

        # ---------------------------------------------
        # GO TO POINT B
        # ---------------------------------------------
        if current_target == "B":

            print("Going to POINT B")

            nav.goToPose(point_b)

            while not nav.isTaskComplete():

                feedback = nav.getFeedback()

                if feedback:
                    print("Moving to B...")

            result = nav.getResult()

            print(f"Result: {result}")

            current_target = "A"

        # ---------------------------------------------
        # GO TO POINT A
        # ---------------------------------------------
        else:

            print("Going to POINT A")

            nav.goToPose(point_a)

            while not nav.isTaskComplete():

                feedback = nav.getFeedback()

                if feedback:
                    print("Moving to A...")

            result = nav.getResult()

            print(f"Result: {result}")

            current_target = "B"

    # =========================================================
    # SHUTDOWN
    # =========================================================

    rclpy.shutdown()


if __name__ == "__main__":
    main()
