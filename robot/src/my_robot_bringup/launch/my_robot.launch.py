from launch import LaunchDescription
from launch.actions import RegisterEventHandler, IncludeLaunchDescription
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command

from ament_index_python.packages import get_package_share_path, get_package_share_directory
import os


def generate_launch_description():

    # ==================================================
    # Paths
    # ==================================================
    description_pkg_path = get_package_share_path("my_robot_description")
    bringup_pkg_path = get_package_share_path("my_robot_bringup")

    urdf_xacro_path = os.path.join(
        description_pkg_path,
        "urdf",
        "my_robot.urdf.xacro"
    )

    controller_yaml_path = os.path.join(
        bringup_pkg_path,
        "config",
        "my_robot_controllers.yaml"
    )

    ekf_config = os.path.join(
        bringup_pkg_path,
        "config",
        "ekf.yaml"
    )

    # ==================================================
    # Robot description (xacro → URDF)
    # ==================================================
    robot_description = Command([
        "xacro", " ", urdf_xacro_path
    ])

    # ==================================================
    # robot_state_publisher
    # ==================================================
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description
        }]
    )

    # ==================================================
    # ros2_control controller_manager
    # ==================================================
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[controller_yaml_path],
        remappings=[
            ("~/robot_description", "/robot_description")
        ]
    )

    # ==================================================
    # Controller spawners (WAIT controller_manager)
    # ==================================================
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen"
    )

    diff_drive_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_drive_controller"],
        output="screen"
    )

    imu_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_sensor_broadcaster"],
        output="screen"
    )

    spawn_controllers_after_cm = RegisterEventHandler(
        OnProcessStart(
            target_action=ros2_control_node,
            on_start=[
                joint_state_broadcaster_spawner,
                diff_drive_controller_spawner,
                imu_broadcaster_spawner,
            ],
        )
    )

    twist_node = Node(
        package="my_robot_hardware",
        executable="twist_relay",
        name="twist_relay",
    )

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_config]
    )

    # ==================================================
    # Launch description
    # ==================================================
    return LaunchDescription([
        robot_state_publisher_node,
        ros2_control_node,
        spawn_controllers_after_cm,
        twist_node,
        ekf_node,
    ])
