# System
import time
import sys
import os
import glob
from datetime import datetime
from threading import Thread
import shutil
import json

# ROS
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion, PoseStamped, Pose, Point, PoseArray
from std_msgs.msg import Header, Bool, String, Int32
from sensor_msgs.msg import Image, CompressedImage, JointState, Joy
from builtin_interfaces.msg import Time
from tf2_msgs.msg import TFMessage
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.executors import MultiThreadedExecutor
from transforms3d.euler import quat2euler
from transforms3d.quaternions import mat2quat
from nav_msgs.msg import Path
from transforms3d.euler import quat2euler, euler2quat
from custom_actions.action import Gripper, GraspObject, ArmTagPose, SendPose, SendJointPose, SendBool

# rosbags
from rosbags.rosbag2 import Writer, Reader
from rosbags.serde import serialize_cdr, deserialize_cdr
from rosbags.typesys.types import geometry_msgs__msg__TransformStamped as rbTransformStamped
from rosbags.typesys.types import geometry_msgs__msg__Transform as rbTransform
from rosbags.typesys.types import geometry_msgs__msg__Vector3 as rbVector3
from rosbags.typesys.types import geometry_msgs__msg__Quaternion as rbQuaternion
from rosbags.typesys.types import std_msgs__msg__Header as rbHeader
from rosbags.typesys.types import builtin_interfaces__msg__Time as rbTime

# Boston Dynamics
from bosdyn.client.math_helpers import SE3Pose, Quat

# Spot Driver
from spot_msgs.msg import BatteryStateArray

# PyQt5
# PyQt5
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QGridLayout,QTabWidget,QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QCursor,QImage, QPalette, QBrush
from PyQt5.QtCore import QSize
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QTableWidgetItem


# Cartesian DMP
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from movement_primitives.dmp import CartesianDMP
from movement_primitives.dmp._dmp import dmp_imitate, dmp_open_loop
from movement_primitives.promp import ProMP

import cv2
from cv_bridge import CvBridge
import numpy as np

class HandTracker(Node):
    def __init__(self):
        super().__init__("hand_tracker")

        # Create subscriber for images
        self.image_sub = self.create_subscription(CompressedImage, "/zed2/zed_node/left/image_rect_color/compressed", self.set_image_callback, 10)
        self.image = None

        # GUI Intialization
        self.gui = QMainWindow()
        self.initGUI()
        self.gui.show()
    def initGUI(self):
        self.gui.setWindowTitle("Main Window")
        self.window_height = 800
        self.window_width = 800
        self.gui.resize(self.window_width, self.window_height)

    def set_image_callback(self, msg):
        self.get_logger().info("RECEIVED IMAGE", throttle_duration_sec=3)
        self.image = msg
    
    def update_image_callback(self):

        self.get_logger().info("Running image callback", throttle_duration_sec=3)
        
        if self.image is not None:
            bridge = CvBridge()
            converted_image = np.asarray(bridge.compressed_imgmsg_to_cv2(self.image, 'rgb8'))
            qimage = QtGui.QImage(converted_image, converted_image.shape[1], converted_image.shape[0], QtGui.QImage.Format_RGB888)
            pixmap = QPixmap(qimage)
            pixmap = pixmap.scaled(self.window_height,self.window_width, Qt.KeepAspectRatio)
            label = QtWidgets.QLabel(self.gui)
            label.setGeometry(QtCore.QRect(0, 0, 800,800))
            label.setPixmap(pixmap)
            label.show()
        else:
            self.get_logger().info("TRYING TO DISPLAY IMAGE", throttle_duration_sec=1)

    def run(self, app):

        while True:
            app.processEvents()
            self.update_image_callback()
# Main Function
def main():
    # Initalize ROS
    rclpy.init()

    app = QApplication(sys.argv)

    hand_tracker = HandTracker()

    executor = MultiThreadedExecutor()
    executor.add_node(hand_tracker)

    thread = Thread(target=executor.spin)
    thread.start()

    try:
        hand_tracker.run(app)
    finally:
        executor.shutdown()

if __name__ == "__main__":
    main()