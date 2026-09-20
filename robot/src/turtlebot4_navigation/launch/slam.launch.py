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
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace, SetRemap

from nav2_common.launch import RewrittenYaml


pkg_turtlebot4_navigation = get_package_share_directory(
    'turtlebot4_navigation'
)
pkg_slam_toolbox = get_package_share_directory(
    'slam_toolbox'
)


ARGUMENTS = [
    DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        choices=['true', 'false'],
        description='Use sim time'
    ),

    DeclareLaunchArgument(
        'sync',
        default_value='true',
        choices=['true', 'false'],
        description='Use synchronous SLAM'
    ),

    DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace'
    ),

    DeclareLaunchArgument(
        'autostart',
        default_value='true',
        choices=['true', 'false'],
        description='Automatically startup slam_toolbox'
    ),

    DeclareLaunchArgument(
        'use_lifecycle_manager',
        default_value='false',
        choices=['true', 'false'],
        description='Enable lifecycle manager'
    ),

    # File cấu hình đầy đủ ban đầu
    DeclareLaunchArgument(
        'params',
        default_value=PathJoinSubstitution([
            pkg_turtlebot4_navigation,
            'config',
            'slam.yaml'
        ]),
        description='Base SLAM Toolbox configuration file'
    ),

    # File chỉ chứa các tham số cần ghi đè
    DeclareLaunchArgument(
        'params_override',
        default_value=PathJoinSubstitution([
            pkg_turtlebot4_navigation,
            'config',
            'slam_pi_override.yaml'
        ]),
        description='SLAM parameter override file'
    )
]


def deep_merge(base, override):
    """
    Ghép dictionary theo kiểu đệ quy.

    Khi một khóa xuất hiện ở cả hai file,
    giá trị trong override sẽ được ưu tiên.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value

    return base


def merge_yaml_files(base_file, override_file):
    """
    Đọc file mặc định và file override,
    sau đó tạo một file YAML tạm đã ghép.
    """
    with open(base_file, 'r', encoding='utf-8') as file:
        base_data = yaml.safe_load(file) or {}

    with open(override_file, 'r', encoding='utf-8') as file:
        override_data = yaml.safe_load(file) or {}

    merged_data = deep_merge(base_data, override_data)

    temporary_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        prefix='slam_merged_',
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
    sync = LaunchConfiguration('sync')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    use_lifecycle_manager = LaunchConfiguration(
        'use_lifecycle_manager'
    )

    slam_params = LaunchConfiguration('params')
    slam_params_override = LaunchConfiguration('params_override')

    namespace_str = namespace.perform(context)

    if namespace_str and not namespace_str.startswith('/'):
        namespace_str = '/' + namespace_str

    # Chuyển LaunchConfiguration thành đường dẫn thực
    base_params_path = slam_params.perform(context)
    override_params_path = slam_params_override.perform(context)

    # Ghép:
    # slam.yaml + slam_pi_override.yaml
    merged_params_path = merge_yaml_files(
        base_params_path,
        override_params_path
    )

    launch_slam_sync = PathJoinSubstitution([
        pkg_slam_toolbox,
        'launch',
        'online_sync_launch.py'
    ])

    launch_slam_async = PathJoinSubstitution([
        pkg_slam_toolbox,
        'launch',
        'online_async_launch.py'
    ])

    # Sau khi ghép mới thực hiện namespace và topic rewrite
    rewritten_slam_params = RewrittenYaml(
        source_file=merged_params_path,
        root_key=namespace_str,
        param_rewrites={
            'map_name': namespace_str + '/map',
            'scan_topic': namespace_str + '/scan',
        },
        convert_types=True,
    )

    slam = GroupAction([
        PushRosNamespace(namespace),

        SetRemap('/tf', namespace_str + '/tf'),
        SetRemap('/tf_static', namespace_str + '/tf_static'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                launch_slam_sync
            ),
            launch_arguments=[
                ('use_sim_time', use_sim_time),
                ('autostart', autostart),
                (
                    'use_lifecycle_manager',
                    use_lifecycle_manager
                ),
                (
                    'slam_params_file',
                    rewritten_slam_params
                )
            ],
            condition=IfCondition(sync)
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                launch_slam_async
            ),
            launch_arguments=[
                ('use_sim_time', use_sim_time),
                ('autostart', autostart),
                (
                    'use_lifecycle_manager',
                    use_lifecycle_manager
                ),
                (
                    'slam_params_file',
                    rewritten_slam_params
                )
            ],
            condition=UnlessCondition(sync)
        )
    ])

    return [slam]


def generate_launch_description():
    launch_description = LaunchDescription(ARGUMENTS)

    launch_description.add_action(
        OpaqueFunction(function=launch_setup)
    )

    return launch_description