#!/usr/bin/env python3

import math
import random
import subprocess
import time

import rclpy

from geometry_msgs.msg import Twist
from nav2_simple_commander.robot_navigator import BasicNavigator
from tf2_ros import Buffer, TransformListener

from rclpy.duration import Duration
import os
import os
import signal
import subprocess
from geometry_msgs.msg import PoseStamped

rclpy.init()

navigator = BasicNavigator()

node = navigator

time.sleep(3.0)

navigator.waitUntilNav2Active()

print("NAV2 ACTIVE")


# ============================================================
# RANDOM NAVIGATION
# ============================================================

x = -1.0
y = -0.01

radius = 0.6

goals = [

    (x + radius, y),
    (x, y + radius),
    (x - radius, y),
    (x, y - radius),

]

goal_index = 0

while rclpy.ok():

    gx, gy = goals[goal_index]

    print(f"GOAL: {gx:.2f}, {gy:.2f}")

    goal = PoseStamped()

    goal.header.frame_id = 'map'

    goal.pose.position.x = gx
    goal.pose.position.y = gy

    goal.pose.orientation.w = 1.0

    navigator.goToPose(goal)

    while not navigator.isTaskComplete():

        rclpy.spin_once(node, timeout_sec=0.1)

    print("GOAL DONE")

    goal_index += 1

    if goal_index >= len(goals):
        goal_index = 0

    time.sleep(1.0)








