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


mapper_proc = subprocess.Popen([
    'ros2',
    'run',
    'bumperbot_mapping',
    'mapping_with_known_poses'
],
    preexec_fn=os.setsid
)

time.sleep(5.0)


# ============================================================
# SAVE MAP
# ============================================================

print("SAVE MAP")
mapname="8"
map_dir = f'/home/pi/dev_ws/src/bumperbot_mapping/maps/{mapname}'

os.makedirs(map_dir, exist_ok=True)

subprocess.run([
    'ros2',
    'run',
    'nav2_map_server',
    'map_saver_cli',
    '-f',
    f'{map_dir}/map'
])

time.sleep(2.0)

# ============================================================
# STOP MAPPER
# ============================================================

os.killpg(
    os.getpgid(mapper_proc.pid),
    signal.SIGTERM
)

print("STOP MAPPER")