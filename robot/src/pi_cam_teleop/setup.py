from setuptools import find_packages, setup

package_name = 'pi_cam_teleop'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='duongduccuong30042002@gmail.com',
    description=(
        'Teleop ban phim voi stream Pi Camera va chup anh bang phim c'
    ),
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'pi_cam_teleop = pi_cam_teleop.pi_cam_teleop_node:main',
        ],
    },
)
