from setuptools import setup
import os
from glob import glob

package_name = 'ciis_drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thariqh15',
    maintainer_email='thariqh15@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = ciis_drone.my_node:main',
            'talker = ciis_drone.publisher_member_function:main',
            'listener = ciis_drone.subscriber_member_function:main',
            'offboard_control = ciis_drone.offboard_control:main',
        ],
    },
)
