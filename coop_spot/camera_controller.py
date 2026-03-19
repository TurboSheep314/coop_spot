# System
import sys
from threading import Thread
import math
import time
import random
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


        # Publishers for the MUAB

        self.update_human_reward = self.create_publisher(Bool, "human_reward", 10)
        self.update_robot_reward = self.create_publisher(Bool, "robot_reward", 10)

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

        # Game setup:
        self.recommendations = {1:False, 2:False, 3:False, 4:False}
        self.stored_scores = list()

    def boltzmann_callback(self, msg):
        self.reward_history["Boltzmann"] = msg.data

    def ucb_callback(self, msg):
        self.reward_history["UCB"] = msg.data

    def human_callback(self, msg):
        self.reward_history["Human"] = msg.data


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
        message.transform.translation.x = 0.85
        message.transform.translation.y = -0.045
        message.transform.translation.z = 0.33
        message.transform.rotation = Quaternion()
        # message.transform.rotation.x = -0.028
        # message.transform.rotation.y = 0.639
        # message.transform.rotation.z = 0.033
        # message.transform.rotation.w = 0.768
        message.transform.rotation.x = 0.0382
        message.transform.rotation.y = 0.6080
        message.transform.rotation.z = -0.0581
        message.transform.rotation.w = 0.7909
        message.child_frame_id = "hand"

        # Send the message to robot_controller
        # self.desired_arm_body_pose_pub.publish(message)
        msg = SendPose.Goal()
        msg.pose = message
        self.desired_arm_body_pose_nc_service_client.wait_for_server()
        self.desired_arm_body_pose_nc_service_client.send_goal_async(msg)

    def show_object(self):
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
        message.transform.translation.x = -0.218
        message.transform.translation.y = -0.032
        message.transform.translation.z = 0.661
        message.transform.rotation = Quaternion()
        # message.transform.rotation.x = -0.028
        # message.transform.rotation.y = 0.639
        # message.transform.rotation.z = 0.033
        # message.transform.rotation.w = 0.768
        message.transform.rotation.x = -0.0194
        message.transform.rotation.y = -0.0039
        message.transform.rotation.z = 0.9964
        message.transform.rotation.w = -0.0825
        message.child_frame_id = "hand"

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


        # --- Calibrate Spot Button --- #

        self.cali_button = QtWidgets.QPushButton(self.gui)
        self.cali_button.setGeometry(QtCore.QRect(775, 50, 200, 50))
        self.cali_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.cali_button.setObjectName("cali_button")
        self.cali_button.setText("Calibrate Spot")
        self.cali_button.clicked.connect(self.spot_arm_calibrate_pos)
        self.cali_button.show()

        # --- Stow Arm Button ---
        self.stow_arm_button = QtWidgets.QPushButton(self.gui)
        self.stow_arm_button.setGeometry(QtCore.QRect(1025, 50, 200, 50))
        self.stow_arm_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.stow_arm_button.setObjectName("stow_arm_button")
        self.stow_arm_button.setText("Stow Arm")
        self.stow_arm_button.clicked.connect(self.stow_arm)
        self.stow_arm_button.show()

        # --- Gripper Open Button ---
        self.gripper_open_button = QtWidgets.QPushButton(self.gui)
        self.gripper_open_button.setGeometry(QtCore.QRect(775, 150, 200, 50))
        self.gripper_open_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.gripper_open_button.setObjectName("gripper_open_button")
        self.gripper_open_button.setText("Open Gripper")
        self.gripper_open_button.clicked.connect(self.open_gripper)
        self.gripper_open_button.show()

        # --- Gripper Close Button ---
        self.gripper_close_button = QtWidgets.QPushButton(self.gui)
        self.gripper_close_button.setGeometry(QtCore.QRect(1025, 150, 200, 50))
        self.gripper_close_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.gripper_close_button.setObjectName("gripper_close_button")
        self.gripper_close_button.setText("Close Gripper")
        self.gripper_close_button.clicked.connect(self.close_gripper)
        self.gripper_close_button.show()
        
        # --- Move to Table 1 Button ---
        self.create_button("table_1_button", "Goto Table 1", QtCore.QRect(525, 250, 200, 50), lambda: self.goto_waypoint(1))

        # --- Move to Table 2 Button ---
        self.create_button("table_2_button", "Goto Table 2", QtCore.QRect(775, 250, 200, 50), lambda: self.goto_waypoint(2))

        # --- Move to Table 3 Button ---
        self.create_button("table_3_button", "Goto Table 3", QtCore.QRect(1025, 250, 200, 50), lambda: self.goto_waypoint(3))

        # --- Move to Table 4 Button ---
        self.create_button("table_4_button", "Goto Table 4", QtCore.QRect(1275, 250, 200, 50), lambda: self.goto_waypoint(4))

        # --- Show Object Spot Button ---
        self.create_button("show_obj_button", "Show Object", QtCore.QRect(1750, 50, 200, 50), self.show_object)

        # --- Return Object Button ---
        self.create_button("return_object_button", "Return Object", QtCore.QRect(1750, 150, 200, 50), self.return_obj)

        # --- Return to Spawn Button ---
        self.create_button("spawn_button", "Goto Spawn", QtCore.QRect(1750, 250, 200, 50), lambda: self.goto_waypoint(0))

        # --- Probabilistic Grasp ---
        self.create_button("prob_grasp_button", "Prob Grasp", QtCore.QRect(900, 550, 200, 50), self.prob_grasp)

        # --- Game Flow ---
        self.create_button("recc_door_button", "Travel to Recommended", QtCore.QRect(800, 650, 400, 50), self.travel_to_recommended)
        self.create_button("random_door_button", "Travel to Random Door", QtCore.QRect(800, 750, 400, 50), self.travel_to_random)
        
        self.create_button("accept_button", "Accept", QtCore.QRect(775, 850, 200, 50), self.accepted_obj)
        self.create_button("deny_button", "Decline", QtCore.QRect(1025, 850, 200, 50), self.denied_obj)

    def travel_to_random(self):
        # Choose door randomly
        door = random.choice(list(self.recommendations.keys()))
        self.get_logger().info("Travelling to door " + str(door))
        self.goto_waypoint(door)

    def travel_to_recommended(self):
        remaining_doors = list()
        for key in self.recommendations.keys():
            if not self.recommendations[key]: # Has yet to be explored.
                remaining_doors.append(key)
        
        self.get_logger().info("Unexplored doors: " + str(remaining_doors))

        # Choose door randomly from remaining options.
        door = random.choice(remaining_doors)
        self.get_logger().info("Travelling to door " + str(door))
        self.goto_waypoint(door)

        # Set door as explored
        self.recommendations[door] = True

    def create_button(self, name : str, text : str, dim : QtCore.QRect, func):
        self.temp = QtWidgets.QPushButton(self.gui)
        self.temp.setGeometry(dim)
        self.temp.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.temp.setObjectName(name)
        self.temp.setText(text)
        self.temp.clicked.connect(func)
        self.temp.show()

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

    # Input should be the door, 0 is start!
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
    
    def prob_grasp(self):
        self.get_logger().info("Called Probabilistic Grasp")
        msg_reward = Bool()
        msg_reward.data = True
        self.update_robot_reward.publish(msg_reward)
        
        msg = SendInt.Goal()

        tag_choices = [12, 65]

        # Choose which tag randomly
        msg.data = random.choice(tag_choices)
        
        self.grasp_tag_service_client.wait_for_server()
        
        returned_msg = self.grasp_tag_service_client.send_goal(msg)
        
        if returned_msg.result.success: # If grasp was successful
            self.get_logger().info("Handing off object!")
            self.show_object()
            # Store a score
            self.stored_scores.append(-1) # nonsense value
        else:
            self.get_logger().info("Not handing off object!")
        


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

    def run(self, app):
        """Primary method, called from main function"""

        while True:
            app.processEvents()
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
