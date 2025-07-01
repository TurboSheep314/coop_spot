# System
import sys
from threading import Thread
import math

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

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
            0: {"name": "table", "present": False},
            12: {"name": "powder", "present": False},
        }

        # GUI Intialization
        self.gui = QMainWindow()
        self.initGUI()
        self.gui.show()

    def initGUI(self):
        """Initializes basic GUI elements"""
        self.gui.setWindowTitle("ZED camera")
        self.window_height = 1200
        self.window_width = 1400
        self.gui.resize(self.window_width, self.window_height)
        self.veltext = QtWidgets.QLabel(self.gui)
        self.veltext.setGeometry(QtCore.QRect(100, 0, 800, 200))
        self.proxtext = QtWidgets.QLabel(self.gui)
        self.proxtext.setGeometry(QtCore.QRect(100, 100, 800, 200))

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
        self.get_logger().info(
            f"April tag IDs detected: {apriltag_IDs}", throttle_duration_sec=3
        )

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
        self.get_logger().info(
            f"World state: {current_world_state}",
            throttle_duration_sec=3,
        )

    def hypotenuse(self, p1, p2):
        """Calculates the hypotenuse between two points in 2d space."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def update_image_callback(self):
        """Method is called continuously from a loop, visualizes hand tracking and april tag data"""
        self.get_logger().info("Running image callback", throttle_duration_sec=3)

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
            label.setGeometry(QtCore.QRect(100, 100, 1200, 1200))
            label.setPixmap(pixmap)
            label.show()

            self.veltext.setText("Wrist velocity: " + " ".join(map(str, self.velocity)))
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
