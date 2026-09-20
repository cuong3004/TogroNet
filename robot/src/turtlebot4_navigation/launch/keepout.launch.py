#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mask_yaml = LaunchConfiguration("mask_yaml")

    return LaunchDescription([
        DeclareLaunchArgument(
            "mask_yaml",
            description="Đường dẫn tới file keepout YAML"
        ),

        Node(
            package="nav2_map_server",
            executable="map_server",
            name="keepout_filter_mask_server",
            output="screen",
            parameters=[{
                "use_sim_time": False,
                "frame_id": "map",
                "topic_name": "/keepout_filter_mask",
                "yaml_filename": mask_yaml,
            }],
        ),

        Node(
            package="nav2_map_server",
            executable="costmap_filter_info_server",
            name="keepout_costmap_filter_info_server",
            output="screen",
            parameters=[{
                "use_sim_time": False,
                "type": 0,
                "filter_info_topic": "/keepout_costmap_filter_info",
                "mask_topic": "/keepout_filter_mask",
                "base": 0.0,
                "multiplier": 1.0,
            }],
        ),

        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_keepout",
            output="screen",
            parameters=[{
                "use_sim_time": False,
                "autostart": True,
                "node_names": [
                    "keepout_filter_mask_server",
                    "keepout_costmap_filter_info_server",
                ],
            }],
        ),
    ])