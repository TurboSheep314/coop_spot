from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="coop_spot",
                executable="camera_controller",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="coop_spot",
                executable="robot_controller",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="coop_spot",
                executable="decision_making",
                output="screen",
                emulate_tty=True,
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("zed_wrapper"),
                        "launch/zed_camera.launch.py",
                    )
                ),
                launch_arguments={"camera_model": "zed2"}.items(),
            ),
        ]
    )
