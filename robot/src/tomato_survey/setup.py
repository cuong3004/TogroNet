import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'tomato_survey'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='duongduccuong30042002@gmail.com',
    description=(
        'Pha 1 survey_run (follow_waypoints + chup anh khong dung) va '
        'Pha 2 batch_infer (TogroNet-Stage/Fruit sau khi ve)'
    ),
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'survey_run = nodes.survey_run:main',
            'batch_infer = scripts.batch_infer:main',
        ],
    },
)
