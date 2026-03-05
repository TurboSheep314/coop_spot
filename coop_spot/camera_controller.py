# System
import sys
from threading import Thread
import math
import time
import os

# ROS
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import CompressedImage, Joy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Header, Bool, Float64MultiArray
from geometry_msgs.msg import (
    TransformStamped,
    Transform,
    Vector3,
    Quaternion,
)
from custom_actions.action import (
    Gripper,
    GraspObject,
    ArmTagPose,
    SendPose,
    SendJointPose,
    SendBool,
    SendInt,
)

import pyqtgraph as pg

from builtin_interfaces.msg import Time  # I dont know what this is but we need it

# rosbags

# Boston Dynamics


# Spot Driver


# PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
)
from PyQt5.QtGui import QPixmap

# from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt

# Cartesian DMP

# OpenCV
import cv2
from cv_bridge import CvBridge
import mediapipe as mp

# Misc
import numpy as np
from apriltag import apriltag


# Data Collection
import csv


class CameraProcessNode(Node):
    """
    This node is for anything to do from the ZED Camera.

    Primary purpose is to determine human intention. Currently has a messy handtracking
    solution and solid april tag detection. The goal is to determine what the human is
    reaching for, if we have two objects, A and B, and the human hand seems to be
    approaching the april tag for object A, we can assume that the human's intention
    is to pick up Object A.
    """

    def __init__(self):
        super().__init__("camera_node")
        # PUBLISHERS
        self.grasp_tag_service_client = ActionClient(
            self, SendInt, "desired_tag_to_grasp"
        )
        
        # Publish what tag to move to
        self.goto_tag_service_client = ActionClient(
            self, SendInt, "desired_tag_to_goto"
        )

        # Publish what waypoint to move to
        self.goto_waypoint_service_client = ActionClient(
            self, SendInt, "desired_waypoint_to_goto"
        )

        # Publish desired arm pose relative to body reference frame
        self.desired_arm_body_pose_service_client = ActionClient(
            self, SendPose, "desired_arm_body_pose"
        )
        self.desired_arm_body_pose_nc_service_client = ActionClient(
            self, SendPose, "desired_arm_body_pose_nc"
        )  # no calibration
        # Publish desired arm joint position
        self.desired_arm_joint_pose_service_client = ActionClient(
            self, SendJointPose, "desired_arm_joint_pose"
        )

        # Publish desired arm pose trajectory relative to tag reference frame
        self.desired_arm_tag_trajectory_service_client = ActionClient(
            self, ArmTagPose, "desired_arm_tag_trajectory"
        )

        # Publish desired arm pose relative to tag reference frame
        self.desired_arm_tag_pose_service_client = ActionClient(
            self, SendPose, "desired_arm_tag_pose"
        )

        # Published object position so robot controller can grasp
        self.grasp_object_service_client = ActionClient(
            self, GraspObject, "grasp_object"
        )

     

        # Publish desired body pose relative to previous body reference frame
        self.desired_body_pose_service_client = ActionClient(
            self, SendPose, "desired_body_pose"
        )

        # Publish desired gripper state
        self.gripper_service_client = ActionClient(
            self, Gripper, "desired_gripper_state"
        )


        self.stow_arm_client = ActionClient(self, SendBool, "stow_arm")

        self.return_obj_client = ActionClient(self, SendBool, "return_obj")
        # publish nearest tag

        # Publishers for the MUAB

        self.update_human_reward = self.create_publisher(Bool, "human_reward", 10)
        self.update_robot_reward = self.create_publisher(Bool, "robot_reward", 10)

        # self.closest_apriltag_pub = self.create_publisher(String, 'closest_tag', 10)

        # Stop/Warn Controller Input
        self.create_subscription(Joy, "/joy", self.controller_input_callback, 1)

        # publish nearest tag
        # self.closest_apriltag_pub = self.create_publisher(String, 'closest_tag', 10)

        # Create subscriber for images
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/zed2/zed_node/left/image_rect_color/compressed",
            self.set_image_callback,
            10,
        )
        self.image = None

        # Hand Tracking Initialization
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.3)
        self.mpDraw = mp.solutions.drawing_utils

        self.bboxes = dict()  # For storing the bounding boxes of all seen april tags

        # Data Initialization
        self.prev_wrist = None
        self.velocity = (0, 0)
        self.closest_apriltag = (-1, math.inf)

        # Data Collection
        csvfile = open("data.csv", "w", newline="")
        self.writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        self.writer.writerow(
            ["Velocity X"]
            + ["Velocity Y"]
            + ["Closest Apriltag ID"]
            + ["Distance between Hand and Apriltag"]
        )
        self.world_state = {
            4: {"name": "bowl", "present": False},
            2: {"name": "bleach", "present": False},
            43: {"name": "sponge", "present": False},
            60: {"name": "table", "present": False},
            12: {"name": "powder", "present": False},
        }

        # Reward history
        self.reward_history = {}
        self.create_subscription(
            Float64MultiArray, "/reward_history_boltzmann", self.boltzmann_callback, 10
        )
        self.create_subscription(
            Float64MultiArray, "/reward_history_ucb", self.ucb_callback, 10
        )
        self.create_subscription(
            Float64MultiArray, "/reward_history_human", self.human_callback, 10
        )

        # GUI Intialization
        self.gui = QMainWindow()
        self.initGUI()
        self.gui.show()

    def boltzmann_callback(self, msg):
        self.reward_history["Boltzmann"] = msg.data

    def ucb_callback(self, msg):
        self.reward_history["UCB"] = msg.data

    def human_callback(self, msg):
        self.reward_history["Human"] = msg.data

    def controller_input_callback(self, msg):
        if msg.buttons[1]:  # B
            if self.stop_on:
                pass
            else:
                self.get_logger().warn("SAVING INTERRUPT IMAGES")
                basepath = "/home/aicaring/AICaring/LIFIdata"
                if self.learned:
                    filename = os.path.join(
                        basepath,
                        str(self.user_ID),
                        str(self.trial_num)
                        + "learned_warn_image_"
                        + str(time.time())
                        + ".png",
                    )
                else:
                    filename = os.path.join(
                        basepath,
                        str(self.user_ID),
                        str(self.trial_num)
                        + "_warn_image_"
                        + str(time.time())
                        + ".png",
                    )
                if len(self.warn_images) > 0:
                    self.save_image(filename, passed_image=self.warn_images[0])
            self.spot_arm_calibrate_pos()
            return
        else:
            self.stop_on = False

        # if msg.buttons[2]: # X
        #     self.warn_images.append(self.image)
        #     if self.warn_on:
        #         #self.get_logger().warn("warn already on")
        #         pass
        #     self.warn_on = True
        # else:
        #     if self.warn_on:
        #         self.get_logger().warn("SAVING WARNING IMAGES")
        #         basepath = "/home/aicaring/AICaring/LIFIdata"
        #         if self.learned:
        #             filename = os.path.join(basepath, str(self.user_ID), str(self.trial_num)+"learned_warn_image_"+str(time.time())+".png")
        #         else:
        #             filename = os.path.join(basepath, str(self.user_ID), str(self.trial_num)+"_warn_image_"+str(time.time())+".png")
        #         if len(self.warn_images) > 0:
        #             self.save_image(filename, passed_image=self.warn_images[0])
        #         self.warn_images = []
        #     self.warn_on = False

    # def study_deploy(self):
    #     self.get_logger().info("Started Study", throttle_duration_sec=3)
    # # stand robot up
    #     #self.sit_stand_spot(stand_flag = True)

    #     # deploy spot to position
    #     #self.spot_body_deploy()

    #     # calibrate spot
    #     self.spot_arm_calibrate_pos()

    def spot_arm_calibrate_pos(self):
        # this moves Spot's arm into the calibrate position

        # ensures Spot's gripper is closed
        self.get_logger().info("CLOSING GRIPPER", throttle_duration_sec=3)
        message = Gripper.Goal()
        message.gripper_state = True
        self.gripper_service_client.wait_for_server()
        self.gripper_service_client.send_goal_async(message)
        self.get_logger().info("Gripper Closed", throttle_duration_sec=3)
        # Create arm pose message
        self.get_logger().info("MOVING ARM", throttle_duration_sec=3)
        message = TransformStamped()
        message.header = Header()
        message.header.stamp = Time()
        t = time.time()
        message.header.stamp.sec = int(t)
        message.header.stamp.nanosec = int((t - int(t)) * 1e9)
        message.header.frame_id = "body"
        message.transform = Transform()
        message.transform.translation = Vector3()
        message.transform.translation.x = 0.4
        message.transform.translation.y = 0.0
        message.transform.translation.z = 0.4
        message.transform.rotation = Quaternion()
        # message.transform.rotation.x = -0.028
        # message.transform.rotation.y = 0.639
        # message.transform.rotation.z = 0.033
        # message.transform.rotation.w = 0.768
        message.transform.rotation.x = 0.0
        message.transform.rotation.y = 0.216
        message.transform.rotation.z = 0.0
        message.transform.rotation.w = 0.976
        message.child_frame_id = "hand"

        # Send the message to robot_controller
        # self.desired_arm_body_pose_pub.publish(message)
        msg = SendPose.Goal()
        msg.pose = message
        self.desired_arm_body_pose_nc_service_client.wait_for_server()
        self.desired_arm_body_pose_nc_service_client.send_goal_async(msg)

    def spot_hand_off_pos(self):
        # this moves Spot's arm into the hand off position

        # ensures Spot's gripper is closed
        # self.get_logger().info("CLOSING GRIPPER", throttle_duration_sec=3)
        # message = Gripper.Goal()
        # message.gripper_state = True
        # self.gripper_service_client.wait_for_server()
        # self.gripper_service_client.send_goal_async(message)
        # self.get_logger().info("Gripper Closed", throttle_duration_sec=3)
        # Create arm pose message
        self.get_logger().info("MOVING ARM", throttle_duration_sec=3)
        message = TransformStamped()
        message.header = Header()
        message.header.stamp = Time()
        t = time.time()
        message.header.stamp.sec = int(t)
        message.header.stamp.nanosec = int((t - int(t)) * 1e9)
        message.header.frame_id = "body"
        message.transform = Transform()
        message.transform.translation = Vector3()
        message.transform.translation.x = 0.6
        message.transform.translation.y = -0.5
        message.transform.translation.z = 0.7
        message.transform.rotation = Quaternion()
        # message.transform.rotation.x = -0.028
        # message.transform.rotation.y = 0.639
        # message.transform.rotation.z = 0.033
        # message.transform.rotation.w = 0.768
        message.transform.rotation.x = 0.153
        message.transform.rotation.y = 0.153
        message.transform.rotation.z = -0.691
        message.transform.rotation.w = 0.691
        message.child_frame_id = "hand"

        # Send the message to robot_controller
        # self.desired_arm_body_pose_pub.publish(message)
        msg = SendPose.Goal()
        msg.pose = message
        self.desired_arm_body_pose_nc_service_client.wait_for_server()
        self.desired_arm_body_pose_nc_service_client.send_goal_async(msg)

    def point_left_right(self, direction):
        # Points Spot's arm towards `direction`

        # ensures Spot's gripper is closed
        # self.get_logger().info("CLOSING GRIPPER", throttle_duration_sec=3)
        # message = Gripper.Goal()
        # message.gripper_state = True
        # self.gripper_service_client.wait_for_server()
        # self.gripper_service_client.send_goal_async(message)
        # self.get_logger().info("Gripper Closed", throttle_duration_sec=3)
        # Create arm pose message
        self.get_logger().info("MOVING ARM", throttle_duration_sec=3)
        message = TransformStamped()
        message.header = Header()
        message.header.stamp = Time()
        t = time.time()
        message.header.stamp.sec = int(t)
        message.header.stamp.nanosec = int((t - int(t)) * 1e9)
        message.header.frame_id = "body"
        message.child_frame_id = "hand"
        message.transform = Transform()
        message.transform.translation = Vector3()
        message.transform.rotation = Quaternion()
        if direction == "left":
            message.transform.translation.x = 0.5
            message.transform.translation.y = 0.5
            message.transform.translation.z = 0.5
            message.transform.rotation.x = 0.0
            message.transform.rotation.y = 0.216
            message.transform.rotation.z = 0.0
            message.transform.rotation.w = 0.976

        elif direction == "right":
            message.transform.translation.x = 0.5
            message.transform.translation.y = -0.5
            message.transform.translation.z = 0.5
            message.transform.rotation.x = 0.153
            message.transform.rotation.y = 0.153
            message.transform.rotation.z = 0.691
            message.transform.rotation.w = 0.691
        else:
            self.get_logger().error(
                "The direction can only be `left` or `right`, no action taken"
            )
            return

        # Send the message to robot_controller
        # self.desired_arm_body_pose_pub.publish(message)
        msg = SendPose.Goal()
        msg.pose = message
        self.desired_arm_body_pose_nc_service_client.wait_for_server()
        self.desired_arm_body_pose_nc_service_client.send_goal_async(msg)

    def initGUI(self):
        """Initializes basic GUI elements"""
        self.gui.setWindowTitle("ZED camera")
        self.window_height = 1400
        self.window_width = 2000
        self.gui.resize(self.window_width, self.window_height)

        # # Layouts to manage the GUI with grids
        # self.bottom_row = QtWidgets.QWidget()
        # self.top_row = = QtWidgets.QWidget()
        # bottom_row_layout = QtWidgets.QGridLayout()
        # self.bottom_row.setLayout(bottom_row_layout)
        # top_row_layout = QtWidgets.QGridLayout()
        # self.top_row.setLayout(top_row_layout)

        self.veltext = QtWidgets.QLabel(self.gui)
        self.veltext.setGeometry(QtCore.QRect(100, 500, 1600, 200))
        self.proxtext = QtWidgets.QLabel(self.gui)
        self.proxtext.setGeometry(QtCore.QRect(100, 50, 800, 200))

        # --- Calibrate Spot Button --- #

        self.cali_button = QtWidgets.QPushButton(self.gui)
        self.cali_button.setGeometry(QtCore.QRect(700, 50, 200, 50))
        self.cali_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.cali_button.setObjectName("cali_button")
        self.cali_button.setText("Calibrate Spot")
        self.cali_button.clicked.connect(self.spot_arm_calibrate_pos)
        self.cali_button.show()

        # --- Grasp Water Spot Button ---
        self.grasp_water_button = QtWidgets.QPushButton(self.gui)
        self.grasp_water_button.setGeometry(QtCore.QRect(700, 150, 200, 50))
        self.grasp_water_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.grasp_water_button.setObjectName("grasp_object_button")
        self.grasp_water_button.setText("Grasp Water")
        self.grasp_water_button.clicked.connect(self.grasp_water)
        self.grasp_water_button.show()

        # --- Grasp Soda Spot Button ---
        self.grasp_soda_button = QtWidgets.QPushButton(self.gui)
        self.grasp_soda_button.setGeometry(QtCore.QRect(700, 250, 200, 50))
        self.grasp_soda_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.grasp_soda_button.setObjectName("grasp_soda_button")
        self.grasp_soda_button.setText("Grasp Soda")
        self.grasp_soda_button.clicked.connect(self.grasp_soda)
        self.grasp_soda_button.show()

        # --- Gripper Open Button ---
        self.gripper_open_button = QtWidgets.QPushButton(self.gui)
        self.gripper_open_button.setGeometry(QtCore.QRect(1000, 50, 200, 50))
        self.gripper_open_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.gripper_open_button.setObjectName("gripper_open_button")
        self.gripper_open_button.setText("Open Gripper")
        self.gripper_open_button.clicked.connect(self.open_gripper)
        self.gripper_open_button.show()

        # --- Gripper Close Button ---
        self.gripper_close_button = QtWidgets.QPushButton(self.gui)
        self.gripper_close_button.setGeometry(QtCore.QRect(1000, 150, 200, 50))
        self.gripper_close_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.gripper_close_button.setObjectName("gripper_close_button")
        self.gripper_close_button.setText("Close Gripper")
        self.gripper_close_button.clicked.connect(self.close_gripper)
        self.gripper_close_button.show()

        # --- Stow Arm Button ---
        self.stow_arm_button = QtWidgets.QPushButton(self.gui)
        self.stow_arm_button.setGeometry(QtCore.QRect(1000, 250, 200, 50))
        self.stow_arm_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.stow_arm_button.setObjectName("stow_arm_button")
        self.stow_arm_button.setText("Stow Arm")
        self.stow_arm_button.clicked.connect(self.stow_arm)
        self.stow_arm_button.show()

        # --- Hand off object Spot Button ---
        self.hand_off_button = QtWidgets.QPushButton(self.gui)
        self.hand_off_button.setGeometry(QtCore.QRect(1300, 50, 200, 50))
        self.hand_off_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.hand_off_button.setObjectName("hand_off_button")
        self.hand_off_button.setText("Hand Object")
        self.hand_off_button.clicked.connect(self.spot_hand_off_pos)
        self.hand_off_button.show()

        # --- Point direction buttons ---
        self.point_direction_button = QtWidgets.QPushButton(self.gui)
        self.point_direction_button.setGeometry(QtCore.QRect(1600, 50, 200, 50))
        self.point_direction_button.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        )
        self.point_direction_button.setObjectName("point_left_button")
        self.point_direction_button.setText("Point left")
        self.point_direction_button.clicked.connect(
            lambda: self.point_left_right(direction="left")
        )
        self.point_direction_button.show()

        self.point_direction_button = QtWidgets.QPushButton(self.gui)
        self.point_direction_button.setGeometry(QtCore.QRect(1600, 150, 200, 50))
        self.point_direction_button.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        )
        self.point_direction_button.setObjectName("point_right_button")
        self.point_direction_button.setText("Point right")
        self.point_direction_button.clicked.connect(
            lambda: self.point_left_right(direction="right")
        )
        self.point_direction_button.show()

        # --- Move to Table 1 Button ---

        self.move_to_table_button = QtWidgets.QPushButton(self.gui)
        self.move_to_table_button.setGeometry(QtCore.QRect(1600, 250, 200, 50))
        self.move_to_table_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.move_to_table_button.setObjectName("table_1_button")
        self.move_to_table_button.setText("Goto Table (Waypoint)")
        self.move_to_table_button.clicked.connect(lambda: self.goto_waypoint(0)) # Number should be door-1
        self.move_to_table_button.show()

        # --- Move to Table 2 Button ---

        self.move_to_table_button = QtWidgets.QPushButton(self.gui)
        self.move_to_table_button.setGeometry(QtCore.QRect(1600, 350, 200, 50))
        self.move_to_table_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.move_to_table_button.setObjectName("table_2_button")
        self.move_to_table_button.setText("Goto Table (id:4)")
        self.move_to_table_button.clicked.connect(lambda: self.goto_tag(4)) # Number should be april tag number
        self.move_to_table_button.show()

        # --- Move to Custom Table Button ---

        self.move_to_table_button = QtWidgets.QPushButton(self.gui)
        self.move_to_table_button.setGeometry(QtCore.QRect(1600, 450, 200, 50))
        self.move_to_table_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.move_to_table_button.setObjectName("table_custom_button")
        self.move_to_table_button.setText("Goto Table")
        self.move_to_table_button.clicked.connect(lambda: self.goto_tag(int(self.table_input.text()))) # Number should be april tag number
        self.move_to_table_button.show()

        self.table_input = QtWidgets.QLineEdit(self.gui)
        self.table_input.setGeometry(QtCore.QRect(1600, 550, 200, 50))
        self.table_input.setPlaceholderText("Enter ID")
        self.table_input.setObjectName("table_input")
        self.int_validator = QtGui.QIntValidator(0, 100, self.gui)
        self.table_input.setValidator(self.int_validator)
        self.move_to_table_button.show()

        # --- Return Object Button ---
        self.return_object_button = QtWidgets.QPushButton(self.gui)
        self.return_object_button.setGeometry(QtCore.QRect(1300, 150, 200, 50))
        self.return_object_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.return_object_button.setObjectName("return_object_button")
        self.return_object_button.setText("Return Object")
        self.return_object_button.clicked.connect(self.return_obj)
        self.return_object_button.show()

        # --- Accept Button ---
        self.accept_button = QtWidgets.QPushButton(self.gui)
        self.accept_button.setGeometry(QtCore.QRect(1300, 250, 200, 50))
        self.accept_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.accept_button.setObjectName("accept_button")
        self.accept_button.setText("Accept")
        self.accept_button.clicked.connect(self.accepted_obj)
        self.accept_button.show()

        # -- Deny Button ---
        self.deny_button = QtWidgets.QPushButton(self.gui)
        self.deny_button.setGeometry(QtCore.QRect(1300, 350, 200, 50))
        self.deny_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.deny_button.setObjectName("deny_button")
        self.deny_button.setText("Deny")
        self.deny_button.clicked.connect(self.denied_obj)
        self.deny_button.show()

    def update_plot(self):
        if self.reward_history.get("UCB"):
            num_rounds = len(self.reward_history["UCB"])
            self.plot_graph = pg.PlotWidget(self.gui)
            # self.setCentralWidget(self.plot_graph)
            self.plot_graph.setBackground("#212946")
            self.plot_graph.setTitle(
                "Bandit Strategies: Boltzmann vs UCB1", color="w", size="14pt"
            )
            styles = {"color": "white", "font-size": "30px"}
            self.plot_graph.setLabel("left", "Cumulative Reward", **styles)
            self.plot_graph.setLabel("bottom", "Round", **styles)
            self.plot_graph.addLegend()
            self.plot_graph.showGrid(x=True, y=True)
            # self.plot_graph.setYRange(20, 40)
            # self.layout.addWidget(self.plot_graph, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows
            self.plot_graph.setGeometry(QtCore.QRect(1400, 500, 1200, 800))
            # cm = pg.colormap.get('Set2', source='matplotlib')
            cm = [
                "#08F7FE",  # teal/cyan
                "#FE53BB",  # pink
                "#F5D300",  # yellow
                "#00ff41",  # matrix green
            ]
            self.Round = np.arange(0, num_rounds)
            self.cum_reward = {}
            self.line = {}

            for idx, label in enumerate(self.reward_history):
                pen = pg.mkPen(color=cm[idx])
                self.cum_reward[label] = np.cumsum(self.reward_history[label])

                if label == "Human" and len(self.cum_reward["Human"]) < len(
                    self.cum_reward["UCB"]
                ):
                    return
                if label == "Human":
                    self.veltext.setText(
                        "UCB: "
                        + str(np.sum(self.cum_reward["UCB"]))
                        + " Boltzmann: "
                        + str(np.sum(self.cum_reward["Boltzmann"]))
                        + " Human: "
                        + str(np.sum(self.cum_reward["Human"]))
                    )
                    self.veltext.show()

                # self.line[label] = self.plot_graph.plot(
                #     self.Round,
                #     self.cum_reward[label],
                #     name=label.capitalize(),
                #     pen=pen,
                #     # symbol="o",
                #     # symbolSize=5,
                #     # symbolBrush="b",
                # )

            # self.plot_graph.show()
            # self.get_logger().info(f"Shapes H:{self.line['Human'].shape}, R:{self.line['UCB'].shape},B:{self.line['Boltzmann'].shape}")

    def accepted_obj(self):
        self.get_logger().info("Human accepted object")
        msg = Bool()
        msg.data = True
        self.update_human_reward.publish(msg)

    def denied_obj(self):
        self.get_logger().info("Human denied object")
        msg = Bool()
        msg.data = False
        self.update_human_reward.publish(msg)

    def return_obj(self):
        self.get_logger().info("Returning Object")
        msg = SendBool.Goal()
        msg.state = True
        self.return_obj_client.wait_for_server()
        return self.return_obj_client.send_goal(msg)

    def open_gripper(self):
        message = Gripper.Goal()
        message.gripper_state = False
        self.gripper_service_client.wait_for_server()
        return self.gripper_service_client.send_goal(message)

    def close_gripper(self):
        message = Gripper.Goal()
        message.gripper_state = True
        self.gripper_service_client.wait_for_server()
        return self.gripper_service_client.send_goal(message)

    def stow_arm(self):
        msg = SendBool.Goal()
        msg.state = True
        self.stow_arm_client.wait_for_server()
        return self.stow_arm_client.send_goal(msg)

    # Input should be the door
    def goto_waypoint(self, door_number):
        self.get_logger().info("Attempting to move to door " + str(door_number) + ".")
        msg = SendInt.Goal()
        msg.data = door_number
        self.goto_waypoint_service_client.wait_for_server()
        self.goto_waypoint_service_client.send_goal_async(msg)
        
    # Input should be the value of the april tag spot will move to
    def goto_tag(self, tag_number):
        self.get_logger().info("Attempting to move to April Tag " + str(tag_number) + ".")
        msg = SendInt.Goal()
        msg.data = tag_number
        self.goto_tag_service_client.wait_for_server()
        self.goto_tag_service_client.send_goal_async(msg)
        
    def grasp_water(self):
        self.get_logger().info("Called Grasp Water")
        msg_reward = Bool()
        msg_reward.data = True
        self.update_robot_reward.publish(msg_reward)

        msg = SendInt.Goal()
        msg.data = 58
        self.grasp_tag_service_client.wait_for_server()
        self.grasp_tag_service_client.send_goal_async(msg)

    def grasp_soda(
        self,
    ):  # right now this should just do the same thing as soda, just testing button
        self.get_logger().info("Called Grasp Soda")
        msg_reward = Bool()
        msg_reward.data = False
        self.update_robot_reward.publish(msg_reward)

        msg = SendInt.Goal()
        msg.data = 65
        self.grasp_tag_service_client.wait_for_server()
        self.grasp_tag_service_client.send_goal_async(msg)

    def set_image_callback(self, msg):
        """Callback for each new image captured by ZED Camera"""
        self.get_logger().info("RECEIVED IMAGE", throttle_duration_sec=3)
        self.image = msg

    def detect_fiducials_in_image(self):
        """Detects all apriltags from image and returns their bounding boxes"""
        bridge = CvBridge()
        img = bridge.compressed_imgmsg_to_cv2(self.image, "rgb8")
        # Convert image data from RGB to Greyscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # required for april tag detection

        # Create apriltag detector
        detector = apriltag(family="tag36h11")  # tag36h11 is the family of tags we use

        # Get detections on greyscale image
        detections = detector.detect(img)
        apriltag_IDs = []
        for i in range(len(detections)):
            # Get bounding box for current detection
            bbox = detections[i]["lb-rb-rt-lt"]
            bbox_id = detections[i]["id"]
            apriltag_IDs.append(bbox_id)

            # Add current detection to dictionary of april tag bounding boxes
            self.bboxes[bbox_id] = bbox
        #self.get_logger().info(
        #    f"April tag IDs detected: {apriltag_IDs}", throttle_duration_sec=3
        #)

        # Update the world state
        for state in self.world_state:
            # Reset the world state
            self.world_state[state]["present"] = False
        for id in apriltag_IDs:
            if id in self.world_state.keys():
                self.world_state[id]["present"] = True
        current_world_state = []
        for state in self.world_state:
            if self.world_state[state]["present"]:
                current_world_state.append(self.world_state[state]["name"])
        #self.get_logger().info(
        #    f"World state: {current_world_state}",
        #    throttle_duration_sec=3,
        #)

    def hypotenuse(self, p1, p2):
        """Calculates the hypotenuse between two points in 2d space."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def update_image_callback(self):
        """Method is called continuously from a loop, visualizes hand tracking and april tag data"""
        #self.get_logger().info("Running image callback", throttle_duration_sec=3)

        if self.image is not None:
            bridge = CvBridge()
            img = bridge.compressed_imgmsg_to_cv2(self.image, "rgb8")

            # Detect fiducials in image (april tags)
            self.detect_fiducials_in_image()
            # Draw april tag locations
            if self.bboxes:
                i = 0
                for coords in self.bboxes.values():
                    while i < len(coords):
                        cx, cy = int(coords[i][0]), int(coords[i][1])
                        cx2, cy2 = (
                            int(coords[(i + 1) % len(coords)][0]),
                            int(coords[(i + 1) % len(coords)][1]),
                        )

                        cv2.line(img, (cx, cy), (cx2, cy2), (255, 255, 0), 5)
                        i += 1
                    i = 0

            # Prepare image for Handtracking
            img = cv2.flip(img, 1)  # For handtracking, the image must be flipped
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # and in RGB

            # Hand Tracking (Copied from mediapipe documentation, except for velocity)
            wristlm = 0
            results = self.hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        print(id, cx, cy)
                        if id == 4:
                            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                        if id == 8:
                            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                        if id == 12:
                            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                        if id == 16:
                            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                        if id == 20:
                            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                        # Velocity of Wrist
                        if id == 0:
                            wristlm = lm
                            if self.prev_wrist is not None:
                                # Number of images per second is 15, will make it more accurate in the future,
                                # but this is consistent for the first couple minutes of the cameras
                                self.velocity = (
                                    (cx - self.prev_wrist[0]) / 15,
                                    (cy - self.prev_wrist[1]) / 15,
                                )
                            self.prev_wrist = (cx, cy)

                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

                # Draw distance markers + Calculate closest april tag
                if self.bboxes:
                    # Reset closest apriltag for current check
                    self.closest_apriltag = (-1, math.inf)
                    h, w, c = img.shape
                    handcx, handcy = (
                        int(wristlm.x * w),
                        int(wristlm.y * h),
                    )  # Converts positional data to the (x,y) of the image.
                    cv2.circle(img, (handcx, handcy), 10, (0, 0, 255), cv2.FILLED)

                    i = 0
                    for id, coords in self.bboxes.items():  # For each april tag
                        # id is the ID of the april tag
                        # coords is the corners of the april tag's bounding box

                        avg_cx, avg_cy = 0, 0
                        # Adds each corner together and averages it, resulting in a point at the center of the april tag.
                        while i < len(coords):
                            cx, cy = int(w - coords[i][0] - 1), int(coords[i][1])
                            avg_cx += cx
                            avg_cy += cy
                            i += 1
                        avg_cx = int(avg_cx / len(coords))
                        avg_cy = int(avg_cy / len(coords))

                        # Draws a line from wrist to april tag.
                        cv2.line(
                            img, (avg_cx, avg_cy), (handcx, handcy), (255, 0, 0), 5
                        )

                        # Distance is the wrist to the center of april tag
                        dist = self.hypotenuse((avg_cx, avg_cy), (handcx, handcy))

                        if dist < self.closest_apriltag[1]:
                            self.closest_apriltag = (id, dist)
                        i = 0

                # Data Collection
                self.writer.writerow(
                    list(map(str, self.velocity))
                    + list(map(str, self.closest_apriltag))
                )

            if self.bboxes:
                # Draw the names of the objects (i.e. world state)
                for id, coords in self.bboxes.items():  # For each april tag
                    # self.get_logger().info(
                    #     f"\nBounding boxes coordinates:\n{coords}",
                    #     throttle_duration_sec=3,
                    # )
                    height, width, _ = img.shape
                    cx, cy = int(width - coords[0][0] - 1), int(coords[0][1])
                    # rect_width = int(len(self.world_state[id]["name"]) * 1.5)
                    # rect_x, rect_y, rect_w, rect_h = cx, cy, rect_width, 50
                    rect_w = 100
                    rect_h = 50
                    # cv2.rectangle(
                    #     img=img,
                    #     pt1=(rect_x, rect_y),
                    #     pt2=(rect_x + rect_w, rect_y + rect_h),
                    #     color=(0, 0, 0),
                    #     thickness=-1,
                    # )
                    if id in self.world_state:
                        cv2.putText(
                            img=img,
                            text=self.world_state[id]["name"],
                            org=(cx + int(rect_w / 10), cy + int(rect_h / 1.5)),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1.0,
                            color=(171, 39, 79),
                            thickness=2,
                            lineType=cv2.FILLED,
                        )
                    else:
                        self.get_logger().error(
                            f"ID #{id} not in world_state", throttle_duration_sec=3
                        )

            converted_image = np.asarray(img)

            qimage = QtGui.QImage(
                converted_image,
                converted_image.shape[1],
                converted_image.shape[0],
                QtGui.QImage.Format_RGB888,
            )
            pixmap = QPixmap(qimage)
            pixmap = pixmap.scaled(
                self.window_height, self.window_width, Qt.KeepAspectRatio
            )
            label = QtWidgets.QLabel(self.gui)
            label.setGeometry(QtCore.QRect(100, 300, 1200, 1200))
            label.setPixmap(pixmap)
            label.show()

            # self.veltext.setText("Wrist velocity: " + " ".join(map(str, self.velocity)))
            self.proxtext.setText(
                "Closest Apriltag: " + " ".join(map(str, self.closest_apriltag))
            )

            self.veltext.show()
            self.proxtext.show()
        else:
            self.get_logger().info("TRYING TO DISPLAY IMAGE", throttle_duration_sec=1)
        # publish closest tag ID
        # if self.closest_tag is not None:
        #    msg= String()
        #    msg.data = "Closest tag is None"
        #    self.closest_apriltag_pub.publish(msg)
        # else:
        #    msg= String()
        #    msg.data = f"CLosest Tag ID: {self.closest_apriltag[0]}"
        #    self.closest_apriltag_pub.publish(msg)

    def run(self, app):
        """Primary method, called from main function"""

        while True:
            app.processEvents()
            self.update_image_callback()
            self.update_plot()
        self.get_logger().info("QUITTING APP")
        app.quit()
        return


# Main Function
def main():
    # Initalize ROS    # Initalize ROS
    rclpy.init()

    app = QApplication(sys.argv)

    camera_node = CameraProcessNode()

    executor = MultiThreadedExecutor()
    executor.add_node(camera_node)

    thread = Thread(target=executor.spin)
    thread.start()

    try:
        camera_node.run(app)
    except KeyboardInterrupt:
        print("Keyboard Interrupt called!")
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
