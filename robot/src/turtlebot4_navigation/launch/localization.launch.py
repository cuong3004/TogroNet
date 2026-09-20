import os
import tempfile
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import PushRosNamespace


pkg_turtlebot4_navigation = get_package_share_directory(
    'turtlebot4_navigation'
)

pkg_nav2_bringup = get_package_share_directory(
    'turtlebot4_navigation'
)


ARGUMENTS = [
    DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        choices=['true', 'false'],
        description='Use sim time'
    ),

    DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace'
    ),

    # File cấu hình localization đầy đủ
    DeclareLaunchArgument(
        'params',
        default_value=PathJoinSubstitution([
            pkg_turtlebot4_navigation,
            'config',
            'localization.yaml'
        ]),
        description='Base localization parameter file'
    ),

    # File chỉ chứa các tham số ghi đè
    DeclareLaunchArgument(
        'params_override_file',
        default_value=PathJoinSubstitution([
            pkg_turtlebot4_navigation,
            'config',
            'localization_pi_override.yaml'
        ]),
        description='Localization parameter override file'
    ),

    DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([
            pkg_turtlebot4_navigation,
            'maps',
            'warehouse.yaml'
        ]),
        description='Full path to map YAML file'
    )
]


def deep_merge(base, override):
    """
    Ghép dictionary đệ quy.

    Nếu một khóa xuất hiện trong cả hai file,
    giá trị trong override sẽ được ưu tiên.
    """
    for key, override_value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(override_value, dict)
        ):
            deep_merge(base[key], override_value)
        else:
            base[key] = override_value

    return base


def merge_yaml_files(base_file, override_file):
    """
    Ghép localization.yaml với file override,
    sau đó tạo file YAML tạm.
    """
    with open(base_file, 'r', encoding='utf-8') as file:
        base_data = yaml.safe_load(file) or {}

    with open(override_file, 'r', encoding='utf-8') as file:
        override_data = yaml.safe_load(file) or {}

    merged_data = deep_merge(base_data, override_data)

    temporary_file = tempfile.NamedTemporaryFile(
        mode='w',
        prefix='localization_merged_',
        suffix='.yaml',
        delete=False,
        encoding='utf-8'
    )

    yaml.safe_dump(
        merged_data,
        temporary_file,
        default_flow_style=False,
        sort_keys=False
    )

    temporary_file.close()

    return temporary_file.name


def launch_setup(context, *args, **kwargs):
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_file = LaunchConfiguration('map')

    params_file = LaunchConfiguration('params')
    params_override_file = LaunchConfiguration(
        'params_override_file'
    )

    # Chuyển LaunchConfiguration thành đường dẫn thực
    base_params_path = params_file.perform(context)
    override_params_path = params_override_file.perform(context)

    if not os.path.isfile(base_params_path):
        raise FileNotFoundError(
            f'Không tìm thấy file localization gốc: '
            f'{base_params_path}'
        )

    if not os.path.isfile(override_params_path):
        raise FileNotFoundError(
            f'Không tìm thấy file localization override: '
            f'{override_params_path}'
        )

    # Override được ghép sau nên có mức ưu tiên cao hơn
    merged_params_path = merge_yaml_files(
        base_params_path,
        override_params_path
    )

    localization_launch = PathJoinSubstitution([
        pkg_nav2_bringup,
        'launch',
        'localization_launch.py'
    ])
    print("localization_launchlocalization_launchlocalization_launchlocalization_launch")
    localization = GroupAction([
        PushRosNamespace(namespace),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                localization_launch
            ),
            launch_arguments={
                'namespace': namespace,
                'map': map_file,
                'use_sim_time': use_sim_time,

                # Truyền file đã ghép
                'params_file': merged_params_path,

                # Giữ giống cách chạy trước
                'use_composition': 'False'
            }.items()
        )
    ])

    return [localization]


def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)

    ld.add_action(
        OpaqueFunction(function=launch_setup)
    )

    return ld
