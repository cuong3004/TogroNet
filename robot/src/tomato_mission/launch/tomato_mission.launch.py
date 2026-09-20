from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    pkg = get_package_share_directory('tomato_mission')

    plant_config = os.path.join(pkg,'config','plants.yaml')

    mission_config = os.path.join(pkg,'config','mission.yaml')

    return LaunchDescription([

        Node(
            package='tomato_mission',
            executable='mission_manager',
            name='mission_manager',
            parameters=[plant_config, mission_config]
        ),

        Node(
            package='tomato_mission',
            executable='capture_node',
            name='capture_node'
        ),

        Node(
            package='tomato_mission',
            executable='detector_node',
            name='detector_node'
        ),

        Node(
            package='tomato_mission',
            executable='result_logger',
            name='result_logger'
        )

    ])