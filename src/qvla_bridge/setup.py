from setuptools import setup
from glob import glob
import os

package_name = 'qvla_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    extras_require={
        'test': ['pytest'],
    },
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Bridge node connecting OpenVLA with Go2 robot control',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bridge_node = qvla_bridge.bridge_node:main',
        ],
    },
) 