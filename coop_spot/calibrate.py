import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Transform, Vector3, Quaternion, Twist
import tf2_ros
from tf_transformations import euler_from_quaternion
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import time

# from coop_spot.camera_controller import CameraProcessNode

from PyQt5.QtWidgets import (
    QApplication,
)
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
import sys

# from apriltag import apriltag
import json
from pathlib import Path


class calib(Node):
    def __init__(self):
        super().__init__("calibrate")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.1, self.run)

        self.T = []
        self.R = []

        self.BASE_FRAME = "tag36h11:60"
        self.CAM_FRAME = "zed2_base_link"

        self.FRAME_COUNT = 10

        # self.counter = 0

    def run(self):
        if self.tf_buffer.can_transform(
            self.BASE_FRAME, self.CAM_FRAME, rclpy.time.Time()
        ):
            msg = self.tf_buffer.lookup_transform(
                self.BASE_FRAME, self.CAM_FRAME, rclpy.time.Time()
            )

            # self.get_logger().info(str(msg))

            trans = [
                msg.transform.translation.x * -1,
                msg.transform.translation.y * -1,
                msg.transform.translation.z * -1,
            ]
            rot = [
                msg.transform.rotation.x,
                msg.transform.rotation.y,
                msg.transform.rotation.z,
                msg.transform.rotation.w,
            ]

            rot = euler_from_quaternion(rot)

            roll_new = rot[0] + np.pi
            pitch_new = rot[1] * -1
            yaw_new = rot[2] + np.pi
            self.get_logger().info("RPY BELOW")
            self.get_logger().info(
                f"roll,pitch, yaw is ['{roll_new:.4f}', '{pitch_new:.4f}', '{yaw_new:.4f}']"
            )

            self.T.append(trans)
            self.R.append(rot)
            self.FRAME_COUNT -= 1
            # self.get_logger().info(f"Found valid transform!")

        # self.get_logger().info(f"{self.counter}")
        # self.counter += 1

        if self.FRAME_COUNT < 0:
            trans = np.array(self.T).mean(axis=0)
            rot = np.array(self.R).mean(axis=0)

            self.get_logger().info(
                f"transform is ['{trans[0]:.4f}', '{trans[1]:.4f}', '{trans[2]:.4f}'], ['{rot[0]:.4f}', '{rot[1]:.4f}', '{rot[2]:.4f}']"
            )

            json_data = {
                "tag36h11:map": {
                    "v1": trans[0],
                    "v2": trans[1],
                    "v3": trans[2],
                    "q1": rot[0],
                    "q2": rot[1],
                    "q3": rot[2],
                }
            }
            saved_json = Path(
                "/home/aicaring/ros2_dev/src/coop_spot/coop_spot/objects_data.json"
            )
            with open(saved_json, "w") as file:
                json.dump(json_data, file, indent=4)

        # else:
        # self.get_logger().warn(f"I AM NOT WORKING BECAUSE MY CODE SUCKS")


# def main():
#     rclpy.init()

#     app = QApplication(sys.argv)
#     camera_node = CameraProcessNode()
#     executor = MultiThreadedExecutor()
#     executor.add_node(camera_node)
#     thread = Thread(target=executor.spin)
#     thread.start()
#     try:
#         camera_node.run(app)
#     except KeyboardInterrupt:
#         print("Keyboard Interrupt called!")
#     finally:
#         executor.shutdown()


#     c = calib()
#     rclpy.spin(c)
def main():
    rclpy.init()

    app = QApplication(sys.argv)

    # temp_node = Node("temp_logger_node")
    # temp_node.get_logger().warn("NODES STARTED")

    # Create both nodes
    # camera_node = CameraProcessNode()
    calib_node = calib()

    # Add both to multi-threaded executor
    # executor = MultiThreadedExecutor()
    # executor.add_node(camera_node)
    # executor.add_node(calib_node)

    # Start ROS executor in a thread
    # ros_thread = Thread(target=executor.spin, daemon=True)
    # ros_thread.start()

    try:
        rclpy.spin(calib_node)
    except KeyboardInterrupt:
        print("Keyboard Interrupt called!")
    finally:
        # executor.shutdown()
        # camera_node.destroy_node()
        calib_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
