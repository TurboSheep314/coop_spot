# System
import time
import sys
from threading import Thread
import math

# ROS
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from rclpy.executors import MultiThreadedExecutor

# rosbags

# Boston Dynamics
from bosdyn import geometry
from bosdyn.api import (
    geometry_pb2,
    image_pb2,
    trajectory_pb2,
    world_object_pb2,
)
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_id import RobotIdClient
from bosdyn.client.power import PowerClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    blocking_stand,
)
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    get_vision_tform_body,
)
from bosdyn.client.math_helpers import Quat
import bosdyn.client.util
from bosdyn.client.docking import (
    DockingClient,
    blocking_undock,
    get_dock_id,
)
from bosdyn.client.arm_surface_contact import ArmSurfaceContactClient

# Spot Driver

# PyQt5
from PyQt5.QtWidgets import (
    QApplication,
)
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


# Cartesian DMP

# OpenCV
import cv2

# Misc
import numpy as np
from apriltag import apriltag
from PIL import Image


# Data Collection


class RobotController(Node):
    """
    This node is for anything ran off of Spot.

    Currently only used for Apriltag detection, but the goal is to have
    Spot detect april tags and pick up objects based on the human's intention.
    If we have two objects, A and B, and the human starts reaching for object A,
    spot will read that intention, and using Object B's april tag, pick it up.
    """

    def __init__(self):
        super().__init__("robot_controller")

        # Stop robot command
        self.create_subscription(Bool, "/stop_robot_flag", self.stop_robot_callback, 1)

        self.setup_robot()

        # Dictionary mapping camera source to it's latest image taken.
        self._image = dict()

        # List of all possible camera sources.
        self._source_names = [
            src.name
            for src in self._image_client.list_image_sources()
            if (
                src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL
                and "depth" not in src.name
            )
        ]

        self.timer = self.create_timer(0.05, self.run)

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_client.get_robot_state()

    @property
    def image(self):
        """Return the current image associated with each source name."""
        return self._image

    @property
    def image_sources_list(self):
        """Return the list of camera sources."""
        return self._source_names

    def populate_source_dict(self):
        """Fills dictionary of the most recently computed camera extrinsics with the camera source.
        The initial boolean indicates if the extrinsics guess should be used."""
        camera_to_extrinsics_guess = dict()
        for src in self._source_names:
            # Dictionary values: use_extrinsics_guess bool, (rotation vector, translation vector) tuple.
            camera_to_extrinsics_guess[src] = (False, (None, None))
        return camera_to_extrinsics_guess

    def run(self):
        """Primary method that is called."""
        self.get_logger().info("Attempt running", throttle_duration_sec=3)
        detected_fiducial = False
        fiducial_rt_world = None

        # Detect the april tag in the images from Spot using the apriltag library.
        bboxes, source_name = self.image_to_bounding_box()
        if bboxes:
            self._previous_source = source_name
            (tvec, _, source_name) = self.pixel_coords_to_camera_coords(
                bboxes, self._intrinsics, source_name
            )
            vision_tform_fiducial_position = self.compute_fiducial_in_world_frame(tvec)
            fiducial_rt_world = geometry_pb2.Vec3(
                x=vision_tform_fiducial_position[0],
                y=vision_tform_fiducial_position[1],
                z=vision_tform_fiducial_position[2],
            )
            detected_fiducial = True
        if detected_fiducial:
            # Go to the tag and stop within a certain distance
            # self.go_to_tag(fiducial_rt_world)
            self.get_logger().info("Fiducial found!", throttle_duration_sec=3)
        else:
            self.get_logger().info("No fiducials found", throttle_duration_sec=3)

        # Power off at the conclusion of the example.
        if self._powered_on:
            self.power_off()

    def stop_robot_callback(self, msg: Bool):
        """Activated when the /stop_robot_flag is called"""
        self.get_logger().warn("SETTING ROBOT FLAG")
        self.stop_robot_flag = msg.data

        if self.stop_robot_flag:
            if self.stop_on:
                self.get_logger().warn("ALREADY STOPPED")
            else:
                self.get_logger().error("STOPPING ROBOT")
                stop_command = RobotCommandBuilder.stop_command()
                self._robot_command_client.robot_command(stop_command)

                stop_time = time.time()
                self.save_state_data(stop_time)

                self.get_logger().info("DONE STOPPING MOVEMENT")
            self.stop_on = True
        else:
            self.stop_on = False

    def get_fiducial_objects(self):
        """Get all fiducials that Spot detects with its perception system."""
        # Get all fiducial objects (an object of a specific type).
        request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        fiducial_objects = self._world_object_client.list_world_objects(
            object_type=request_fiducials
        ).world_objects
        self.get_logger().info(fiducial_objects)
        if len(fiducial_objects) > 0:
            # Return the first detected fiducial.
            return fiducial_objects[0]
        # Return none if no fiducials are found.
        return None

    def power_on(self):
        """Power on the robot."""
        self._robot.power_on()
        self._powered_on = True
        print(f"Powered On {self._robot.is_powered_on()}")

    def power_off(self):
        """Power off the robot."""
        self._robot.power_off()
        print(f"Powered Off {not self._robot.is_powered_on()}")

    def image_to_bounding_box(self):
        """Determine which camera source has a fiducial.
        Return the bounding box of the first detected fiducial."""
        # Iterate through all five camera sources to check for a fiducial
        for i in range(len(self._source_names) + 1):
            # Get the image from the source camera.
            if i == 0:
                if self._previous_source is not None:
                    # Prioritize the camera the fiducial was last detected in.
                    source_name = self._previous_source
                else:
                    continue
            elif self._source_names[i - 1] == self._previous_source:
                continue
            else:
                source_name = self._source_names[i - 1]

            img_req = build_image_request(
                source_name,
                quality_percent=100,
                image_format=image_pb2.Image.FORMAT_RAW,
            )
            image_response = self._image_client.get_image([img_req])
            self._camera_tform_body = get_a_tform_b(
                image_response[0].shot.transforms_snapshot,
                image_response[0].shot.frame_name_image_sensor,
                BODY_FRAME_NAME,
            )
            self._body_tform_world = get_a_tform_b(
                image_response[0].shot.transforms_snapshot,
                BODY_FRAME_NAME,
                VISION_FRAME_NAME,
            )

            # Camera intrinsics for the given source camera.
            self._intrinsics = image_response[0].source.pinhole.intrinsics
            width = image_response[0].shot.image.cols
            height = image_response[0].shot.image.rows

            # detect given fiducial in image and return the bounding box of it
            bboxes = self.detect_fiducial_in_image(
                image_response[0].shot.image, (width, height), source_name
            )
            if bboxes:
                print(f"Found bounding box for {source_name}")
                return bboxes, source_name
            else:
                self._tag_not_located = True
                print(f"Failed to find bounding box for {source_name}")
        return [], None

    def detect_fiducial_in_image(self, image, dim, source_name):
        """Detect the fiducial within a single image and return its bounding box."""
        image_grey = np.array(
            Image.frombytes(
                "P", (int(dim[0]), int(dim[1])), data=image.data, decoder_name="raw"
            )
        )

        # Rotate each image such that it is upright
        image_grey = self.rotate_image(image_grey, source_name)

        # Make the image greyscale to use bounding box detections
        detector = apriltag(family="tag36h11")
        detections = detector.detect(image_grey)

        bboxes = []
        for i in range(len(detections)):
            # Draw the bounding box detection in the image.
            bbox = detections[i]["lb-rb-rt-lt"]
            cv2.polylines(image_grey, [np.int32(bbox)], True, (0, 0, 0), 2)
            bboxes.append(bbox)

        self._image[source_name] = image_grey
        return bboxes

    def bbox_to_image_object_pts(self, bbox):
        """Determine the object points and image points for the bounding box.
        The origin in object coordinates = top left corner of the fiducial.
        Order both points sets following: (TL,TR, BL, BR)"""
        fiducial_height_and_width = 146  # mm
        obj_pts = np.array(
            [
                [0, 0],
                [fiducial_height_and_width, 0],
                [0, fiducial_height_and_width],
                [fiducial_height_and_width, fiducial_height_and_width],
            ],
            dtype=np.float32,
        )
        # insert a 0 as the third coordinate (xyz)
        obj_points = np.insert(obj_pts, 2, 0, axis=1)

        # ['lb-rb-rt-lt']
        img_pts = np.array(
            [
                [bbox[3][0], bbox[3][1]],
                [bbox[2][0], bbox[2][1]],
                [bbox[0][0], bbox[0][1]],
                [bbox[1][0], bbox[1][1]],
            ],
            dtype=np.float32,
        )
        return obj_points, img_pts

    def pixel_coords_to_camera_coords(self, bbox, intrinsics, source_name):
        """Compute transformation of 2d pixel coordinates to 3d camera coordinates."""
        camera = self.make_camera_matrix(intrinsics)
        # Track a triplet of (translation vector, rotation vector, camera source name)
        best_bbox = (None, None, source_name)
        # The best bounding box is considered the closest to the robot body.
        closest_dist = float("inf")
        for i in range(len(bbox)):
            obj_points, img_points = self.bbox_to_image_object_pts(bbox[i])
            if self._camera_to_extrinsics_guess[source_name][0]:
                # initialize the position estimate with the previous extrinsics solution
                # then iteratively solve for new position
                old_rvec, old_tvec = self._camera_to_extrinsics_guess[source_name][1]
                _, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    img_points,
                    camera,
                    np.zeros((5, 1)),
                    old_rvec,
                    old_tvec,
                    True,
                    cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                # Determine current extrinsic solution for the tag.
                _, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points, camera, np.zeros((5, 1))
                )

            # Save extrinsics results to help speed up next attempts to locate bounding box in
            # the same camera source.
            self._camera_to_extrinsics_guess[source_name] = (True, (rvec, tvec))

            dist = (
                math.sqrt(
                    float(tvec[0][0]) ** 2
                    + float(tvec[1][0]) ** 2
                    + float(tvec[2][0]) ** 2
                )
                / 1000.0
            )
            if dist < closest_dist:
                closest_dist = dist
                best_bbox = (tvec, rvec, source_name)

        # Flag indicating if the best april tag been found/located
        self._tag_not_located = best_bbox[0] is None and best_bbox[1] is None
        return best_bbox

    def compute_fiducial_in_world_frame(self, tvec):
        """Transform the tag position from camera coordinates to world coordinates."""
        fiducial_rt_camera_frame = np.array(
            [
                float(tvec[0][0]) / 1000.0,
                float(tvec[1][0]) / 1000.0,
                float(tvec[2][0]) / 1000.0,
            ]
        )
        body_tform_fiducial = (self._camera_tform_body.inverse()).transform_point(
            fiducial_rt_camera_frame[0],
            fiducial_rt_camera_frame[1],
            fiducial_rt_camera_frame[2],
        )
        fiducial_rt_world = self._body_tform_world.inverse().transform_point(
            body_tform_fiducial[0], body_tform_fiducial[1], body_tform_fiducial[2]
        )
        return fiducial_rt_world

    def go_to_tag(self, fiducial_rt_world):
        """Use the position of the april tag in vision world frame and command the robot."""
        # Compute the go-to point (offset by .5m from the fiducial position) and the heading at
        # this point.
        self._current_tag_world_pose, self._angle_desired = self.offset_tag_pose(
            fiducial_rt_world, 0.0
        )

        # Command the robot to go to the tag in kinematic odometry frame
        tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=self._current_tag_world_pose[0],
            goal_y=self._current_tag_world_pose[1],
            goal_heading=self._angle_desired,
            frame_name=VISION_FRAME_NAME,
            body_height=0.0,
            locomotion_hint=spot_command_pb2.HINT_AUTO,
        )
        end_time = 5.0
        if self._powered_on:
            # Issue the command to the robot
            self._robot_command_client.robot_command(
                lease=None, command=tag_cmd, end_time_secs=time.time() + end_time
            )
            # #Feedback to check and wait until the robot is in the desired position or timeout
            start_time = time.time()
            current_time = time.time()
            while not self.final_state() and current_time - start_time < end_time:
                time.sleep(0.25)
                current_time = time.time()
        return

    def final_state(self):
        """Check if the current robot state is within range of the fiducial position."""
        robot_state = get_vision_tform_body(
            self.robot_state.kinematic_state.transforms_snapshot
        )
        robot_angle = robot_state.rot.to_yaw()
        if self._current_tag_world_pose.size != 0:
            x_dist = abs(self._current_tag_world_pose[0] - robot_state.x)
            y_dist = abs(self._current_tag_world_pose[1] - robot_state.y)
            angle = abs(self._angle_desired - robot_angle)
            if (
                (x_dist < self._x_eps)
                and (y_dist < self._y_eps)
                and (angle < self._angle_eps)
            ):
                return True
        return False

    def get_desired_angle(self, xhat):
        """Compute heading based on the vector from robot to object."""
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, xhat)
        mat = np.array([xhat, yhat, zhat]).transpose()
        return Quat.from_matrix(mat).to_yaw()

    def offset_tag_pose(self, object_rt_world, dist_margin=1.0):
        """Offset the go-to location of the fiducial and compute the desired heading."""
        robot_rt_world = get_vision_tform_body(
            self.robot_state.kinematic_state.transforms_snapshot
        )
        robot_to_object_ewrt_world = np.array(
            [
                object_rt_world.x - robot_rt_world.x,
                object_rt_world.y - robot_rt_world.y,
                0,
            ]
        )
        robot_to_object_ewrt_world_norm = robot_to_object_ewrt_world / np.linalg.norm(
            robot_to_object_ewrt_world
        )
        heading = self.get_desired_angle(robot_to_object_ewrt_world_norm)
        goto_rt_world = np.array(
            [
                object_rt_world.x - robot_to_object_ewrt_world_norm[0] * dist_margin,
                object_rt_world.y - robot_to_object_ewrt_world_norm[1] * dist_margin,
            ]
        )
        return goto_rt_world, heading

    @staticmethod
    def set_default_body_control():
        """Set default body control params to current body position"""
        footprint_R_body = geometry.EulerZXY()
        position = geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0)
        rotation = footprint_R_body.to_quaternion()
        pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)
        point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
        traj = trajectory_pb2.SE3Trajectory(points=[point])
        return spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)

    @staticmethod
    def rotate_image(image, source_name):
        """Rotate the image so that it is always displayed upright."""
        if source_name == "frontleft_fisheye_image":
            image = cv2.rotate(image, rotateCode=0)
        elif source_name == "right_fisheye_image":
            image = cv2.rotate(image, rotateCode=1)
        elif source_name == "frontright_fisheye_image":
            image = cv2.rotate(image, rotateCode=0)
        return image

    @staticmethod
    def make_camera_matrix(ints):
        """Transform the ImageResponse proto intrinsics into a camera matrix."""
        camera_matrix = np.array(
            [
                [ints.focal_length.x, ints.skew.x, ints.principal_point.x],
                [ints.skew.y, ints.focal_length.y, ints.principal_point.y],
                [0, 0, 1],
            ]
        )
        return camera_matrix

    def setup_robot(self):  # Taken from user_study
        # Create robot object
        self._robot = create_standard_sdk("robot_controller").create_robot(
            "192.168.80.3"
        )

        # Authenticate robot
        bosdyn.client.util.authenticate(self._robot)

        # Start time sync server
        self._robot.start_time_sync()

        # Ensure robot not estopped
        while self._robot.is_estopped():
            self.get_logger().warning(
                "Waiting for robot estop...", throttle_duration_sec=1
            )

        # Store robot id
        self._robot_id = self._robot.ensure_client(
            RobotIdClient.default_service_name
        ).get_id(timeout=10)

        # Setup clients
        self._lease_client = self._robot.ensure_client(LeaseClient.default_service_name)
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)
        self._image_client = self._robot.ensure_client(ImageClient.default_service_name)
        self._robot_state_client = self._robot.ensure_client(
            RobotStateClient.default_service_name
        )
        self._robot_command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name
        )
        self._world_object_client = self._robot.ensure_client(
            WorldObjectClient.default_service_name
        )
        self._docking_client = self._robot.ensure_channel(
            DockingClient.default_service_name
        )
        self._arm_surface_contact_client = self._robot.ensure_client(
            ArmSurfaceContactClient.default_service_name
        )

        # Setup variables
        self._max_x_vel = 0.4  # m/s
        self._max_y_vel = 0.4  # m/s
        self._max_ang_vel = 0.3  # rad/s
        self._max_arm_lin_vel = 0.4  # m/s
        self._max_arm_ang_vel = 0.3  # rad/s
        self._max_arm_accel = 0.3  # valid range (0,20]
        self._max_arm_joint_vel = 1  # rad/s
        self._max_gripper_vel = 1  # rad/s
        self._max_grasp_accel = 0.3  # rad/s
        self._max_grasp_ang_vel = 0.3  # rad/s

        # Epsilon distance between robot and desired go-to point
        self._x_eps = 0.05
        self._y_eps = 0.05
        self._ang_eps = 0.075

        # Setup global objects
        self._hand_camera_intrinsics = None
        self._hand_camera_tform_body = None
        self._hand_camera_tform_odom = None
        self._hand_camera_tform_vision = None
        self._hand_camera_to_extrinsics_guess = (False, (None, None))

        # Indicator for if motor power is on.
        self._powered_on = False

        # Counter for the number of iterations completed.
        self._attempts = 0

        # Maximum amount of iterations before powering off the motors.
        self._max_attempts = 100000

        # Camera intrinsics for the current camera source being analyzed.
        self._intrinsics = None

        # Transform from the robot's camera frame to the baselink frame.
        # It is a math_helpers.SE3Pose.
        self._camera_tform_body = None

        # Transform from the robot's baselink to the world frame.
        # It is a math_helpers.SE3Pose.
        self._body_tform_world = None

        # Latest detected fiducial's position in the world.
        self._current_tag_world_pose = np.array([])

        # Heading angle based on the camera source which detected the fiducial.
        self._angle_desired = None

        # Dictionary mapping camera source to it's latest image taken.
        self._image = dict()

        # List of all possible camera sources.
        self._source_names = [
            src.name
            for src in self._image_client.list_image_sources()
            if (
                src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL
                and "depth" not in src.name
            )
        ]
        print(self._source_names)

        # Dictionary mapping camera source to previously computed extrinsics.
        self._camera_to_extrinsics_guess = self.populate_source_dict()

        # Camera source which a bounding box was last detected in.
        self._previous_source = None

        self.get_logger().info("LEASES: " + str(self._lease_client.list_leases()))

        # Try to take lease
        try:
            self._lease_client.take()
            self.get_logger().info("LEASE WAS TAKEN")
        except:
            self.get_logger().warn("LEASE COULD NOT BE TAKEN")

        # Create lease
        self._lease_keep_alive = LeaseKeepAlive(
            lease_client=self._lease_client, must_acquire=True, return_at_exit=True
        ).__enter__()

        # Wait for time sync
        self._robot.time_sync.wait_for_sync()

        # Wait for motors to power on
        self._robot.power_on()

        dock_id = get_dock_id(self._robot)
        if dock_id is None:
            self.get_logger().info("Robot is not docked; standing")
        else:
            self.get_logger().info(f"Docked at {dock_id}")
            blocking_undock(self._robot)

        # Stand robot
        blocking_stand(self._robot_command_client)

        self.get_logger().info("Done with robot setup.")


# Main Function
def main():
    # Initalize ROS
    rclpy.init()

    app = QApplication(sys.argv)

    robot_controller = RobotController()

    executor = MultiThreadedExecutor()
    executor.add_node(robot_controller)

    thread = Thread(target=executor.spin)
    thread.start()

    try:
        robot_controller.run()
    except KeyboardInterrupt:
        print("Keyboard Interrupt called!")
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
