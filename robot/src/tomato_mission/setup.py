from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tomato_mission'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),

        # config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='pi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mission_manager = tomato_mission.mission_manager:main',
            'capture_node = tomato_mission.capture_node:main',
            'detector_node = tomato_mission.detector_node:main',
            'result_logger = tomato_mission.result_logger:main',
        ],
    },
)
