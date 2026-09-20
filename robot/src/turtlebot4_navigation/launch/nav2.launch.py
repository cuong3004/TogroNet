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

from launch_ros.actions import PushRosNamespace, SetRemap


pkg_turtlebot4_navigation = get_package_share_directory(
    'turtlebot4_navigation'
)


ARGUMENTS = [
    DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        choices=['true', 'false'],
        description='Use sim time'
    ),

    # File cấu hình đầy đủ
    DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            pkg_turtlebot4_navigation,
            'config',
            'nav2.yaml'
        ]),
        description='Base Nav2 parameter file'
    ),

    # File chỉ chứa các tham số ghi đè
    DeclareLaunchArgument(
        'params_override_file',
        default_value=PathJoinSubstitution([
            pkg_turtlebot4_navigation,
            'config',
            'nav2_pi_override.yaml'
        ]),
        description='Nav2 parameter override file'
    ),

    DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace'
    )
]


def deep_merge(base, override):
    """
    Ghép hai dictionary theo kiểu đệ quy.

    Giá trị trong override sẽ thay thế giá trị trong base
    nếu hai khóa trùng nhau.
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
    Ghép file Nav2 mặc định với file override
    và trả về đường dẫn file YAML tạm.
    """
    with open(base_file, 'r', encoding='utf-8') as file:
        base_data = yaml.safe_load(file) or {}

    with open(override_file, 'r', encoding='utf-8') as file:
        override_data = yaml.safe_load(file) or {}

    merged_data = deep_merge(base_data, override_data)

    temporary_file = tempfile.NamedTemporaryFile(
        mode='w',
        prefix='nav2_merged_',
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
    pkg_nav2_bringup = get_package_share_directory('turtlebot4_navigation')

    params_file = LaunchConfiguration('params_file')
    params_override_file = LaunchConfiguration('params_override_file')

    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')

    namespace_str = namespace.perform(context)

    if namespace_str and not namespace_str.startswith('/'):
        namespace_str = '/' + namespace_str

    # Lấy đường dẫn thực của hai file
    base_params_path = params_file.perform(context)
    override_params_path = params_override_file.perform(context)

    if not os.path.isfile(base_params_path):
        raise FileNotFoundError(
            f'Không tìm thấy file Nav2 mặc định: {base_params_path}'
        )

    if not os.path.isfile(override_params_path):
        raise FileNotFoundError(
            f'Không tìm thấy file Nav2 override: {override_params_path}'
        )

    # File override được ghép sau nên sẽ có ưu tiên cao hơn
    merged_params_path = merge_yaml_files(
        base_params_path,
        override_params_path
    )

    launch_nav2 = PathJoinSubstitution([
        pkg_nav2_bringup,
        'launch',
        'navigation_launch.py'
    ])

    nav2 = GroupAction([
        PushRosNamespace(namespace),

        SetRemap(
            namespace_str + '/global_costmap/scan',
            namespace_str + '/scan'
        ),

        SetRemap(
            namespace_str + '/local_costmap/scan',
            namespace_str + '/scan'
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_nav2),
            launch_arguments=[
                ('use_sim_time', use_sim_time),

                # Truyền file đã ghép vào Nav2
                ('params_file', merged_params_path),

                ('use_composition', 'False'),
                ('namespace', namespace_str)
            ]
        ),
    ])

    return [nav2]


def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(OpaqueFunction(function=launch_setup))

    return ld