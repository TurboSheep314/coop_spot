# System
import os
import time
import signal

# Standard
import numpy as np
import math
import cv2
from PIL import Image
from apriltag import apriltag
import json
from threading import Thread

# ROS
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from geometry_msgs.msg import TransformStamped, PoseStamped
from std_msgs.msg import Bool, String, Int32
from sensor_msgs.msg import JointState
from tf2_ros import TFMessage
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from transforms3d.euler import quat2euler, euler2quat
from custom_actions.action import (
    Gripper,
    GraspObject,
    ArmTagPose,
    SendPose,
    SendJointPose,
    SendBool,
    SendInt,
)
from rclpy.executors import MultiThreadedExecutor


# Boston Dynamics
from bosdyn import geometry
from bosdyn.api import (
    geometry_pb2,
    image_pb2,
    trajectory_pb2,
    world_object_pb2,
    arm_command_pb2,
    robot_command_pb2,
    synchronized_command_pb2,
    gripper_command_pb2,
    arm_surface_contact_pb2,
    arm_surface_contact_service_pb2,
    manipulation_api_pb2,
)
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
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
    blocking_sit,
    block_until_arm_arrives,
    CommandFailedError,
)
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    get_vision_tform_body,
    get_odom_tform_body,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    HAND_FRAME_NAME,
    add_edge_to_tree,
    math_helpers,
)
from bosdyn.client.math_helpers import SE3Pose, Quat
import bosdyn.client.util
from bosdyn.util import seconds_to_duration
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.docking import (
    DockingClient,
    blocking_dock_robot,
    blocking_undock,
    get_dock_id,
)
from bosdyn.client.arm_surface_contact import ArmSurfaceContactClient


# Spot Driver

# # PyQt5
# from PyQt5.QtWidgets import (
#     QApplication,
# )

# from PyQt5.QtGui import *

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

        # Subscriptions/Flags

        # Listen to desired arm pose relative to the body reference frame
        self.desired_arm_body_pose_service = ActionServer(
            self, SendPose, "desired_arm_body_pose", self.desired_arm_body_pose_callback
        )
        self.desired_arm_body_pose_nc_service = ActionServer(
            self,
            SendPose,
            "desired_arm_body_pose_nc",
            self.desired_arm_body_pose_nc_callback,
        )  # no calibration

        # Listen to desired arm joint pose
        self.desired_arm_joint_pose_service = ActionServer(
            self,
            SendJointPose,
            "desired_arm_joint_pose",
            self.desired_arm_joint_pose_callback,
        )

        # Listen to desired arm pose relative to the april tag reference frame
        self.arm_tag_pose_service = ActionServer(
            self, SendPose, "desired_arm_tag_pose", self.desired_arm_tag_pose_callback
        )

        # Listen for desired tag to grasp
        self.arm_grasp_tag_service = ActionServer(
            self, SendInt, "desired_tag_to_grasp", self.grasp_tag
        )

        # Listen to desired tag to move to
        self.goto_tag_service = ActionServer(
            self, SendInt, "desired_tag_to_goto", self.goto_tag
        )

        # Publish any transforms
        self.tf_pub = self.create_publisher(TFMessage, "/tf", 1)

        # Create tf buffer and transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Listen to desired gripper state 0 -> open, 1 -> closed
        self.gripper_service = ActionServer(
            self, Gripper, "desired_gripper_state", self.desired_gripper_state_callback
        )

        self.stow_arm_service = ActionServer(self, SendBool, "stow_arm", self.stow_arm)

        self.return_obj_service = ActionServer(
            self, SendBool, "return_obj", self.return_obj
        )

        # Stop robot command
        self.create_subscription(Bool, "/stop_robot_flag", self.stop_robot_callback, 1)
        self.stop_robot_flag = False
        self.stop_on = False

        self.setup_robot()

        # Dictionary mapping camera source to it's latest image taken.
        self.img = dict()

        # Dictionary mapping april tag locations and source
        self.april_tags = dict()

        # List of all possible camera sources.
        self.source_names = [
            src.name
            for src in self.img_client.list_image_sources()
            if (
                src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL
                and "depth" not in src.name
            )
        ]

        # April tag detection
        self.run_timer = self.create_timer(5, self.run)

        # Calibration
        self.calibration_timer = self.create_timer(5, self.calibration_timer_callback)

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self.robot_state_client.get_robot_state()

    @property
    def image(self):
        """Return the current image associated with each source name."""
        return self.img

    @property
    def image_sources_list(self):
        """Return the list of camera sources."""
        return self.source_names

    def populate_source_dict(self):
        """Fills dictionary of the most recently computed camera extrinsics with the camera source.
        The initial boolean indicates if the extrinsics guess should be used."""
        camera_to_extrinsics_guess = dict()
        for src in self.source_names:
            # Dictionary values: use_extrinsics_guess bool, (rotation vector, translation vector) tuple.
            camera_to_extrinsics_guess[src] = (False, (None, None))
        return camera_to_extrinsics_guess

    def calibration_timer_callback(self):
        # Make sure previous transform exists
        try:
            if (
                self.body_tform_reference
                and self.odom_tform_reference
                and self.vision_tform_reference
            ):
                pass
        except:
            return

        # self.get_logger().info(f"Publishing last reference calibration TF.")

        # Create body_tform_reference
        tfm1 = self.SE3Pose_to_TransformStamped(
            self.vision_tform_reference, "vision", "spot_reference"
        )

        # Create map_tform_odom
        tfm2 = self.SE3Pose_to_TransformStamped(
            self.odom_tform_reference.inverse(), "map", "odom"
        )

        # Create map_tform_odom
        # tfm3 = self.SE3Pose_to_TransformStamped(self.vision_tform_reference.inverse(), "map", "vision")

        # robot_state_transforms = self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        # map_tform_odom = get_a_tform_b(robot_state_transforms, "map", ODOM_FRAME_NAME)
        # self.get_logger().warn(str(map_tform_odom))

        # Publish message
        self.tf_pub.publish(TFMessage(transforms=[tfm1, tfm2]))

    def desired_arm_body_pose_nc_callback(self, goal_handle):
        msg = goal_handle.request.pose

        if not self.stop_robot_flag:
            # Create transform from message
            body_tform_hand = SE3Pose(
                x=msg.transform.translation.x,
                y=msg.transform.translation.y,
                z=msg.transform.translation.z,
                rot=Quat(
                    w=msg.transform.rotation.w,
                    x=msg.transform.rotation.x,
                    y=msg.transform.rotation.y,
                    z=msg.transform.rotation.z,
                ),
            )

            # Create arm command from transform
            arm_command = RobotCommandBuilder.arm_pose_command(
                x=body_tform_hand.x,
                y=body_tform_hand.y,
                z=body_tform_hand.z,
                qw=body_tform_hand.rot.w,
                qx=body_tform_hand.rot.x,
                qy=body_tform_hand.rot.y,
                qz=body_tform_hand.rot.z,
                frame_name=BODY_FRAME_NAME,
                seconds=5,
            )
            arm_command.synchronized_command.arm_command.arm_cartesian_command.max_linear_velocity.value = self.max_arm_lin_vel
            arm_command.synchronized_command.arm_command.arm_cartesian_command.max_angular_velocity.value = self.max_arm_ang_vel
            arm_command.synchronized_command.arm_command.arm_cartesian_command.maximum_acceleration.value = self.max_arm_accel

            # Send arm command to robot
            cmd_id = self.robot_command_client.robot_command(arm_command)

            output = self.block_until_arm_arrives_interruptable(
                self.robot_command_client, cmd_id, timeout_sec=5
            )

            self.get_logger().info("CALIBRATE ARM MOVE COMPLETE")

            # message = Bool()
            # message.data = True
            # self.stop_robot_callback(message)

            goal_handle.succeed()
            result = SendPose.Result()
            result.success = output

            return result
        else:
            self.get_logger().error("ROBOT STOPPED - CALIBRATE POS STOPPED")
            result = SendPose.Result()
            result.success = False
            return result

    def desired_arm_body_pose_callback(self, goal_handle):
        msg = goal_handle.request.pose

        if not self.stop_robot_flag:
            # Create transform from message
            body_tform_hand = SE3Pose(
                x=msg.transform.translation.x,
                y=msg.transform.translation.y,
                z=msg.transform.translation.z,
                rot=Quat(
                    w=msg.transform.rotation.w,
                    x=msg.transform.rotation.x,
                    y=msg.transform.rotation.y,
                    z=msg.transform.rotation.z,
                ),
            )

            # Create arm command from transform
            arm_command = RobotCommandBuilder.arm_pose_command(
                x=body_tform_hand.x,
                y=body_tform_hand.y,
                z=body_tform_hand.z,
                qw=body_tform_hand.rot.w,
                qx=body_tform_hand.rot.x,
                qy=body_tform_hand.rot.y,
                qz=body_tform_hand.rot.z,
                frame_name=BODY_FRAME_NAME,
                seconds=5,
            )
            arm_command.synchronized_command.arm_command.arm_cartesian_command.max_linear_velocity.value = self.max_arm_lin_vel
            arm_command.synchronized_command.arm_command.arm_cartesian_command.max_angular_velocity.value = self.max_arm_ang_vel
            arm_command.synchronized_command.arm_command.arm_cartesian_command.maximum_acceleration.value = self.max_arm_accel

            # Send arm command to robot
            cmd_id = self.robot_command_client.robot_command(arm_command)

            output = self.block_until_arm_arrives_interruptable(
                self.robot_command_client, cmd_id, timeout_sec=5
            )

            self.get_logger().info("CALIBRATE ARM MOVE COMPLETE")
            goal_handle.succeed()
            result = SendPose.Result()
            result.success = output

            for i in range(3):
                self.calibrate_spot_to_reference()
                time.sleep(0.5)

            return result
        else:
            self.get_logger().error("ROBOT STOPPED - CALIBRATE STOPPED")
            result = SendPose.Result()
            result.success = False
            return result

    def desired_arm_tag_pose_callback(self, goal_handle):
        msg = goal_handle.request.pose

        if not self.stop_robot_flag:
            # Create transform from message
            ref_tform_hand = SE3Pose(
                x=msg.transform.translation.x,
                y=msg.transform.translation.y,
                z=msg.transform.translation.z,
                rot=Quat(
                    w=msg.transform.rotation.w,
                    x=msg.transform.rotation.x,
                    y=msg.transform.rotation.y,
                    z=msg.transform.rotation.z,
                ),
            )

            # odom_tform_hand = self.odom_tform_reference * ref_tform_hand
            vision_tform_hand = self.vision_tform_reference * ref_tform_hand

            # body_tform_hand = self.body_tform_reference * ref_tform_hand

            # Create arm command from transform
            arm_command = RobotCommandBuilder.arm_pose_command(
                x=vision_tform_hand.x,
                y=vision_tform_hand.y,
                z=vision_tform_hand.z,
                qw=vision_tform_hand.rot.w,
                qx=vision_tform_hand.rot.x,
                qy=vision_tform_hand.rot.y,
                qz=vision_tform_hand.rot.z,
                frame_name=VISION_FRAME_NAME,
                seconds=5,
            )
            arm_command.synchronized_command.arm_command.arm_cartesian_command.max_linear_velocity.value = self.max_arm_lin_vel
            arm_command.synchronized_command.arm_command.arm_cartesian_command.max_angular_velocity.value = self.max_arm_ang_vel
            arm_command.synchronized_command.arm_command.arm_cartesian_command.maximum_acceleration.value = self.max_arm_accel

            # Send arm command to robot
            cmd_id = self.robot_command_client.robot_command(arm_command)

            output = self.block_until_arm_arrives_interruptable(
                self.robot_command_client, cmd_id, timeout_sec=5
            )

            self.get_logger().info("ARM TAG POSE MOVE COMPLETE")
            goal_handle.succeed()
            result = SendPose.Result()
            result.success = output
            return result
        else:
            self.get_logger().error("ROBOT STOPPED - ARM TAG POSE MOVE STOPPED")
            result = SendPose.Result()
            result.success = False
            return result

    def desired_arm_joint_pose_callback(self, goal_handle):
        msg = goal_handle.request.jointstate

        if not self.stop_robot_flag:
            arm_joint_positions = msg.position
            sh0 = arm_joint_positions[0]
            sh1 = arm_joint_positions[1]
            el0 = arm_joint_positions[2]
            el1 = arm_joint_positions[3]
            wr0 = arm_joint_positions[4]
            wr1 = arm_joint_positions[5]

            traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
                sh0, sh1, el0, el1, wr0, wr1
            )
            arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
            arm_joint_traj.maximum_velocity.value = self.max_arm_joint_vel

            joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(
                trajectory=arm_joint_traj
            )

            arm_command = arm_command_pb2.ArmCommand.Request(
                arm_joint_move_command=joint_move_command
            )
            sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(
                arm_command=arm_command
            )
            arm_sync_robot_cmd = robot_command_pb2.RobotCommand(
                synchronized_command=sync_arm
            )
            command = RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)

            cmd_id = self.robot_command_client.robot_command(command)
            output = self.block_until_arm_arrives_interruptable(
                self.robot_command_client, cmd_id, timeout_sec=5
            )

            self.get_logger().info("STOW ARM COMPLETE")
            goal_handle.succeed()
            result = SendJointPose.Result()
            result.success = output
            return result
        else:
            self.get_logger().error("ROBOT STOPPED - STOW STOPPED")
            result = SendJointPose.Result()
            result.success = False
            return result

    def stow_arm(self, msg):
        if msg.request.state:
            # Build the stow command
            stow = RobotCommandBuilder.arm_stow_command()

            # Issues the command
            stow_command_id = self.robot_command_client.robot_command(stow)

            # Block movement until stowing is complete
            block_until_arm_arrives(self.robot_command_client, stow_command_id, 3.0)

            self.get_logger().info("STOW ARM COMPLETE")
            msg.succeed()
            result = SendBool.Result()
            result.success = True
            return result

        result = SendBool.Result()
        result.sucess = False
        return result

    def desired_gripper_state_callback(self, goal_handle):
        # 0 -> open, 1 -> closed

        self.get_logger().info(
            "GRIPPER MESSAGE: " + str(goal_handle.request.gripper_state)
        )

        if self.stop_robot_flag:
            output = False

            result = Gripper.Result()
            result.success = output

            self.get_logger().error("ROBOT STOPPED - GRIPPER COMMAND STOPPED")
        else:
            # Desired closed
            if goal_handle.request.gripper_state:
                gripper_command = RobotCommandBuilder.claw_gripper_close_command()
            # Desired open
            else:
                gripper_command = RobotCommandBuilder.claw_gripper_open_command()

            # Send gripper command to robot
            cmd_id = self.robot_command_client.robot_command(gripper_command)

            output = self.block_until_gripper_arrives(
                self.robot_command_client, cmd_id, timeout_sec=2
            )

            goal_handle.succeed()

            result = Gripper.Result()
            result.success = output

            self.get_logger().info("GRIPPER CMD RETURN: " + str(output))

        return result

    def run(self):
        """Primary method that is called."""
        self.get_logger().info("Attempt running", throttle_duration_sec=3)
        detected_fiducial = False
        fiducial_rt_world = None

        # Detect the april tag in the images from Spot using the apriltag library.
        bboxes, source_name = self.image_to_bounding_box()
        if bboxes:
            self.previous_source = source_name
            (tvec, _, source_name) = self.pixel_coords_to_camera_coords(
                bboxes, self.intrinsics, source_name
            )
            vision_tform_fiducial_position = self.compute_fiducial_in_world_frame(tvec)
            fiducial_rt_world = geometry_pb2.Vec3(
                x=vision_tform_fiducial_position[0],
                y=vision_tform_fiducial_position[1],
                z=vision_tform_fiducial_position[2],
            )

            self.get_logger().info(
                f"Camera of the fiducial: {source_name}", throttle_duration_sec=3
            )
            if bboxes:
                for id in bboxes:
                    self.get_logger().info(
                        f"Bounding box ID: {id}", throttle_duration_sec=3
                    )
                    if id == 4:
                        self.get_logger().info(
                            "Detected table's fiducial!", throttle_duration_sec=3
                        )
                        detected_fiducial = True
                        break

        # if detected_fiducial:
        #     # Go to the tag and stop within a certain distance
        #     self.get_logger().info(
        #         "Fiducial found, moving towards it...", throttle_duration_sec=3
        #     )
        #     self.go_to_tag(fiducial_rt_world)
        # else:
        #     self.get_logger().info("No fiducials found", throttle_duration_sec=3)

        # # Power off at the conclusion of the example.
        # if self.powered_on:
        #     self.power_off()

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
                self.robot_command_client.robot_command(stop_command)

                self.get_logger().info("DONE STOPPING MOVEMENT")
            self.stop_on = True
        else:
            self.stop_on = False

    def block_until_gripper_arrives(self, command_client, cmd_id, timeout_sec=None):
        # returns when the gripper command is complete or time to complete the command runs out
        # also is interruptable: stops when robot is interrupted

        if timeout_sec is not None:
            start_time = time.time()
            end_time = start_time + timeout_sec
            now = time.time()

        while timeout_sec is None or now < end_time:
            feedback_resp = command_client.robot_command_feedback(cmd_id)
            gripper_feedback = (
                feedback_resp.feedback.synchronized_feedback.gripper_command_feedback
            )

            if gripper_feedback.HasField("claw_gripper_feedback"):
                if (
                    gripper_feedback.claw_gripper_feedback.status
                    == gripper_command_pb2.ClawGripperCommand.Feedback.STATUS_AT_GOAL
                ):
                    return True

            if self.stop_robot_flag:
                self.get_logger().error(
                    "ROBOT STOPPED - BLOCKING GRIPPER COMMAND INTERRUPTED"
                )
                break

            time.sleep(0.1)
            now = time.time()
        return False

    def block_until_arm_arrives_interruptable(
        self, command_client, cmd_id, timeout_sec=None
    ):
        """Helper that blocks until the arm achieves a finishing state for the specific arm command.

        This helper will block and check the feedback for ArmCartesianCommand, GazeCommand,
        ArmJointMoveCommand, NamedArmPositionsCommand, and ArmImpedanceCommand.

        Args:
            command_client: robot command client, used to request feedback
            cmd_id: command ID returned by the robot when the arm movement command was sent.
            timeout_sec: optional number of seconds after which we'll return no matter what
                        the robot's state is.

        Return values:
            True if successfully got to the end of the trajectory, False if the arm stalled or
            the move was canceled (the arm failed to reach the goal). See the proto definitions in
            arm_command.proto for more information about why a trajectory would succeed or fail.
        """
        if timeout_sec is not None:
            start_time = time.time()
            end_time = start_time + timeout_sec
            now = time.time()

        while timeout_sec is None or now < end_time:
            feedback_resp = command_client.robot_command_feedback(cmd_id)
            arm_feedback = (
                feedback_resp.feedback.synchronized_feedback.arm_command_feedback
            )

            if arm_feedback.HasField("arm_cartesian_feedback"):
                if (
                    arm_feedback.arm_cartesian_feedback.status
                    == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE
                ):
                    return True
                elif (
                    arm_feedback.arm_cartesian_feedback.status
                    == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_STALLED
                    or feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status
                    == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_CANCELLED
                ):
                    return False
            elif arm_feedback.HasField("arm_gaze_feedback"):
                if (
                    arm_feedback.arm_gaze_feedback.status
                    == arm_command_pb2.GazeCommand.Feedback.STATUS_TRAJECTORY_COMPLETE
                ):
                    return True
                elif (
                    arm_feedback.arm_gaze_feedback.status
                    == arm_command_pb2.GazeCommand.Feedback.STATUS_TOOL_TRAJECTORY_STALLED
                ):
                    return False
            elif arm_feedback.HasField("arm_joint_move_feedback"):
                if (
                    arm_feedback.arm_joint_move_feedback.status
                    == arm_command_pb2.ArmJointMoveCommand.Feedback.STATUS_COMPLETE
                ):
                    return True
                elif (
                    arm_feedback.arm_joint_move_feedback.status
                    == arm_command_pb2.ArmJointMoveCommand.Feedback.STATUS_STALLED
                ):
                    return False
            elif arm_feedback.HasField("named_arm_position_feedback"):
                if (
                    arm_feedback.named_arm_position_feedback.status
                    == arm_command_pb2.NamedArmPositionsCommand.Feedback.STATUS_COMPLETE
                ):
                    return True
                elif (
                    arm_feedback.named_arm_position_feedback.status
                    == arm_command_pb2.NamedArmPositionsCommand.Feedback.STATUS_STALLED_HOLDING_ITEM
                ):
                    return False
            elif arm_feedback.HasField("arm_impedance_feedback"):
                if (
                    arm_feedback.arm_impedance_feedback.status
                    == arm_command_pb2.ArmImpedanceCommand.Feedback.STATUS_TRAJECTORY_COMPLETE
                ):
                    return True
                elif (
                    arm_feedback.arm_impedance_feedback.status
                    == arm_command_pb2.ArmImpedanceCommand.Feedback.STATUS_TRAJECTORY_STALLED
                ):
                    return False

            if self.stop_robot_flag:
                self.get_logger().error("ROBOT STOPPED - BLOCKING COMMAND INTERRUPTED")
                break

            time.sleep(0.1)
            now = time.time()
        return False

    def get_fiducial_objects(self):
        """Get all fiducials that Spot detects with its perception system."""
        # Get all fiducial objects (an object of a specific type).
        request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        fiducial_objects = self.world_object_client.list_world_objects(
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
        self.robot.power_on()
        self.powered_on = True
        self.get_logger().info(f"Powered On {self.robot.is_powered_on()}")

    def power_off(self):
        """Power off the robot."""
        self.robot.power_off()
        self.get_logger().info(f"Powered Off {not self.robot.is_powered_on()}")

    def image_to_bounding_box(self):
        """Determine which camera source has a fiducial.
        Return the bounding box of the first detected fiducial."""
        bboxes, source_name = [], None
        # Iterate through all five camera sources to check for a fiducial
        for i in range(len(self.source_names) + 1):
            # Get the image from the source camera.
            if i == 0:
                if self.previous_source is not None:
                    # Prioritize the camera the fiducial was last detected in.
                    source_name = self.previous_source
                else:
                    continue
            elif self.source_names[i - 1] == self.previous_source:
                continue
            else:
                source_name = self.source_names[i - 1]

            img_req = build_image_request(
                source_name,
                quality_percent=100,
                image_format=image_pb2.Image.FORMAT_RAW,
            )
            image_response = self.img_client.get_image([img_req])
            self.camera_tform_body = get_a_tform_b(
                image_response[0].shot.transforms_snapshot,
                image_response[0].shot.frame_name_image_sensor,
                BODY_FRAME_NAME,
            )
            self.body_tform_world = get_a_tform_b(
                image_response[0].shot.transforms_snapshot,
                BODY_FRAME_NAME,
                VISION_FRAME_NAME,
            )

            # Camera intrinsics for the given source camera.
            self.intrinsics = image_response[0].source.pinhole.intrinsics
            width = image_response[0].shot.image.cols
            height = image_response[0].shot.image.rows

            # detect given fiducial in image and return the bounding box of it
            bboxes = self.detect_fiducial_in_image(
                image_response[0], (width, height), source_name
            )
            if bboxes:
                self.get_logger().info(f"Found bounding box for {source_name}")
                # return bboxes, source_name
            else:
                self.tag_not_located = True
                self.get_logger().info(f"Failed to find bounding box for {source_name}")
        # return [], None
        return bboxes, source_name

    def detect_fiducial_in_image(self, image, dim, source_name):
        """Detect the fiducial within a single image and return its bounding box."""
        image_grey = np.array(
            Image.frombytes(
                "P",
                (int(dim[0]), int(dim[1])),
                data=image.shot.image.data,
                decoder_name="raw",
            )
        )

        # Make the image greyscale to use bounding box detections
        detector = apriltag(family="tag36h11")
        detections = detector.detect(image_grey)

        self.get_logger().info(f"{detections}")

        bboxes = {}
        for i in range(len(detections)):
            # Draw the bounding box detection in the image.
            bbox = detections[i]["lb-rb-rt-lt"]

            cv2.polylines(image_grey, [np.int32(bbox)], True, (0, 0, 0), 2)
            bbox_id = detections[i]["id"]
            bboxes[bbox_id] = bbox

            self.april_tags[bbox_id] = detections[i]["center"], image

            # bboxes.append(bbox)

        self.img[source_name] = image_grey
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

    def pixel_coords_to_camera_coords(self, bboxes, intrinsics, source_name):
        """Compute transformation of 2d pixel coordinates to 3d camera coordinates."""
        camera = self.make_camera_matrix(intrinsics)
        # Track a triplet of (translation vector, rotation vector, camera source name)
        best_bbox = (None, None, source_name)
        # The best bounding box is considered the closest to the robot body.
        closest_dist = float("inf")
        self.get_logger().info(f"Bounding boxes:\n{bboxes}")
        for id in bboxes:
            self.get_logger().info(f"Bounding box: {bboxes[id]}")
            obj_points, img_points = self.bbox_to_image_object_pts(bboxes[id])
            if self.camera_to_extrinsics_guess[source_name][0]:
                # initialize the position estimate with the previous extrinsics solution
                # then iteratively solve for new position
                old_rvec, old_tvec = self.camera_to_extrinsics_guess[source_name][1]
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
            self.camera_to_extrinsics_guess[source_name] = (True, (rvec, tvec))

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
        self.tag_not_located = best_bbox[0] is None and best_bbox[1] is None
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
        body_tform_fiducial = (self.camera_tform_body.inverse()).transform_point(
            fiducial_rt_camera_frame[0],
            fiducial_rt_camera_frame[1],
            fiducial_rt_camera_frame[2],
        )
        fiducial_rt_world = self.body_tform_world.inverse().transform_point(
            body_tform_fiducial[0], body_tform_fiducial[1], body_tform_fiducial[2]
        )
        return fiducial_rt_world

    def return_obj(self, msg):
        # Returning hand to position after the grasp
        self.get_logger().info("Inside return_obj in robot_controller")
        arm_command = RobotCommandBuilder.arm_pose_command(
            x=self.after_grasp_loc.x,
            y=self.after_grasp_loc.y,
            z=self.after_grasp_loc.z,
            qw=self.after_grasp_loc.rot.w,
            qx=self.after_grasp_loc.rot.x,
            qy=self.after_grasp_loc.rot.y,
            qz=self.after_grasp_loc.rot.z,
            frame_name=VISION_FRAME_NAME,
            seconds=3,
        )
        arm_command.synchronized_command.arm_command.arm_cartesian_command.max_linear_velocity.value = self.max_arm_lin_vel
        arm_command.synchronized_command.arm_command.arm_cartesian_command.max_angular_velocity.value = self.max_grasp_ang_vel
        arm_command.synchronized_command.arm_command.arm_cartesian_command.maximum_acceleration.value = self.max_grasp_accel

        # Send arm command to robot
        cmd_id = self.robot_command_client.robot_command(arm_command)

        output = self.block_until_arm_arrives_interruptable(
            self.robot_command_client, cmd_id, timeout_sec=5
        )

        result = SendBool.Result()
        msg.succeed()
        result.success = True
        return result

    def store_arm_coords(self):
        # Request the robot state
        # check transform from tag to hand (if available)
        self.get_logger().info("Storing arm coordinates...")
        robot_state_transforms = self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        body_tform_hand = get_a_tform_b(
            robot_state_transforms, VISION_FRAME_NAME, HAND_FRAME_NAME
        )

        self.after_grasp_loc = body_tform_hand

        self.get_logger().info(str(self.after_grasp_loc))

    def apply_force_constraint(self, grasp):
        self.get_logger().info("Applying grasp constraints")
        grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

        grasp.grasp_params.grasp_palm_to_fingertip = 0.2
    

    def grasp_tag(self, msg):
        """Use the position of the april tag in vision world frame and command the robot to grasp the attach object"""
        self.get_logger().info("Entered grasp tag")

        tag = self.april_tags[msg.request.data]

        image = tag[1]

        pick_vec = geometry_pb2.Vec2(x=tag[0][0], y=tag[0][1])

        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole,
        )

        # Apply constraints
        self.apply_force_constraint(grasp)

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp
        )

        # Send the request
        cmd_response = self.manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request
        )

        # Get feedback from the robot
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id
            )

            # Send the request
            response = self.manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request
            )

            print(
                f"Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}"
            )

            if (
                response.current_state
                == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
                or response.current_state
                == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
            ):
                break

            time.sleep(0.25)

        self.get_logger().info("Finished grasp.")
        time.sleep(4.0)

        self.store_arm_coords()

        result = SendInt.Result()
        result.success = True
        return result

    def goto_tag(self, msg):
        """Use the position of the april tag in vision world frame and command the robot."""
        # Compute the go-to point (offset by .5m from the fiducial position) and the heading at
        # this point.

        april_tag_id = msg.request.data
        
        fiducial_rt_world = self.get_fiducial_rt_world_from_id(april_tag_id)

        self.current_tag_world_pose, self.angle_desired = self.offset_tag_pose(
            fiducial_rt_world, 0.0
        )

        # Command the robot to go to the tag in kinematic odometry frame
        tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=self.current_tag_world_pose[0],
            goal_y=self.current_tag_world_pose[1],
            goal_heading=self.angle_desired,
            frame_name=VISION_FRAME_NAME,
            body_height=0.0,
            locomotion_hint=spot_command_pb2.HINT_AUTO,
        )
        end_time = 5.0
        if self.powered_on:
            self.get_logger().info(
                f"Goal: {self.current_tag_world_pose[0], self.current_tag_world_pose[1]}"
            )
            # Issue the command to the robot
            self.robot_command_client.robot_command(
                lease=None, command=tag_cmd, end_time_secs=time.time() + end_time
            )
            # #Feedback to check and wait until the robot is in the desired position or timeout
            start_time = time.time()
            current_time = time.time()
            while not self.final_state() and current_time - start_time < end_time:
                time.sleep(0.25)
                current_time = time.time()
        
        result = SendInt.Result()
        result.success = True
        return result

    def final_state(self):
        """Check if the current robot state is within range of the fiducial position."""
        robot_state = get_vision_tform_body(
            self.robot_state.kinematic_state.transforms_snapshot
        )
        robot_angle = robot_state.rot.to_yaw()
        if self.current_tag_world_pose.size != 0:
            x_dist = abs(self.current_tag_world_pose[0] - robot_state.x)
            y_dist = abs(self.current_tag_world_pose[1] - robot_state.y)
            angle = abs(self.angle_desired - robot_angle)
            if (
                (x_dist < self.x_eps)
                and (y_dist < self.y_eps)
                and (angle < self.ang_eps)
            ):
                self.get_logger().info("Robot within range of the fiducial position")
                return True
        self.get_logger().info("Robot NOT within range of the fiducial position")
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
        self.robot = create_standard_sdk("robot_controller").create_robot(
            "192.168.80.3"
        )

        # Authenticate robot
        bosdyn.client.util.authenticate(self.robot)

        # Start time sync server
        self.robot.start_time_sync()

        # Ensure robot not estopped
        while self.robot.is_estopped():
            self.get_logger().warning(
                "Waiting for robot estop...", throttle_duration_sec=1
            )

        # Store robot id
        self.robot_id = self.robot.ensure_client(
            RobotIdClient.default_service_name
        ).get_id(timeout=10)

        # Setup clients

        self.manipulation_api_client = self.robot.ensure_client(
            ManipulationApiClient.default_service_name
        )
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.power_client = self.robot.ensure_client(PowerClient.default_service_name)
        self.img_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.robot_state_client = self.robot.ensure_client(
            RobotStateClient.default_service_name
        )
        self.robot_command_client = self.robot.ensure_client(
            RobotCommandClient.default_service_name
        )
        self.world_object_client = self.robot.ensure_client(
            WorldObjectClient.default_service_name
        )
        self.docking_client = self.robot.ensure_channel(
            DockingClient.default_service_name
        )
        self.arm_surface_contact_client = self.robot.ensure_client(
            ArmSurfaceContactClient.default_service_name
        )

        # Setup variables
        self.max_x_vel = 0.4  # m/s
        self.max_y_vel = 0.4  # m/s
        self.max_ang_vel = 0.3  # rad/s
        self.max_arm_lin_vel = 0.4  # m/s
        self.max_arm_ang_vel = 0.3  # rad/s
        self.max_arm_accel = 0.3  # valid range (0,20]
        self.max_arm_joint_vel = 1  # rad/s
        self.max_gripper_vel = 1  # rad/s
        self.max_grasp_accel = 0.3  # rad/s
        self.max_grasp_ang_vel = 0.3  # rad/s

        # Epsilon distance between robot and desired go-to point
        self.x_eps = 0.05
        self.y_eps = 0.05
        self.ang_eps = 0.075

        # Setup global objects
        self.hand_camera_intrinsics = None
        self.hand_camera_tform_body = None
        self.hand_camera_tform_odom = None
        self.hand_camera_tform_vision = None
        self.hand_camera_to_extrinsics_guess = (False, (None, None))

        # Indicator for if motor power is on.
        self.powered_on = False

        # Counter for the number of iterations completed.
        self.attempts = 0

        # Maximum amount of iterations before powering off the motors.
        self.max_attempts = 100000

        # Camera intrinsics for the current camera source being analyzed.
        self.intrinsics = None

        # Transform from the robot's camera frame to the baselink frame.
        # It is a math_helpers.SE3Pose.
        self.camera_tform_body = None

        # Transform from the robot's baselink to the world frame.
        # It is a math_helpers.SE3Pose.
        self.body_tform_world = None

        # Latest detected fiducial's position in the world.
        self.current_tag_world_pose = np.array([])

        # Heading angle based on the camera source which detected the fiducial.
        self.angle_desired = None

        # Dictionary mapping camera source to it's latest image taken.
        self.img = dict()

        # List of all possible camera sources.
        self.source_names = [
            src.name
            for src in self.img_client.list_image_sources()
            if (
                src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL
                and "depth" not in src.name
            )
        ]
        self.get_logger().info(f"List of camera sources: {self.source_names}")

        # Dictionary mapping camera source to previously computed extrinsics.
        self.camera_to_extrinsics_guess = self.populate_source_dict()

        # Camera source which a bounding box was last detected in.
        self.previous_source = None

        self.get_logger().info("LEASES: " + str(self.lease_client.list_leases()))

        # Try to take lease
        try:
            self.lease_client.take()
            self.get_logger().info("LEASE WAS TAKEN")
        except Exception:
            self.get_logger().warn("LEASE COULD NOT BE TAKEN")

        # Create lease
        self.lease_keep_alive = LeaseKeepAlive(
            lease_client=self.lease_client, must_acquire=True, return_at_exit=True
        ).__enter__()

        # Wait for time sync
        self.robot.time_sync.wait_for_sync()

        # Wait for motors to power on
        self.get_logger().info("Powering on")
        self.power_on()
        # self.robot.power_on()

        dock_id = get_dock_id(self.robot)
        if dock_id is None:
            self.get_logger().info("Robot is not docked; standing")
        else:
            self.get_logger().info(f"Docked at {dock_id}")
            blocking_undock(self.robot)

        # Stand robot
        blocking_stand(self.robot_command_client)

        self.get_logger().info("Done with robot setup.")
    
    def old_image_to_bounding_box(self, target_id):
        """Determine which camera source has a fiducial.
           Return the bounding box of the first detected fiducial."""
        #Iterate through all five camera sources to check for a fiducial
        for i in range(len(self.source_names) + 1):
            # Get the image from the source camera.
            if i == 0:
                if self.previous_source is not None:
                    # Prioritize the camera the fiducial was last detected in.
                    source_name = self.previous_source
                else:
                    continue
            elif self.source_names[i - 1] == self.previous_source:
                continue
            else:
                source_name = self.source_names[i - 1]

            img_req = build_image_request(source_name, quality_percent=100,
                                          image_format=image_pb2.Image.FORMAT_RAW)
            image_response = self.img_client.get_image([img_req])
            self.camera_tform_body = get_a_tform_b(image_response[0].shot.transforms_snapshot,
                                                    image_response[0].shot.frame_name_image_sensor,
                                                    BODY_FRAME_NAME)
            self.body_tform_world = get_a_tform_b(image_response[0].shot.transforms_snapshot,
                                                   BODY_FRAME_NAME, VISION_FRAME_NAME)

            # Camera intrinsics for the given source camera.
            self.intrinsics = image_response[0].source.pinhole.intrinsics
            width = image_response[0].shot.image.cols
            height = image_response[0].shot.image.rows

            # detect given fiducial in image and return the bounding box of it
            bboxes = self.old_detect_fiducial_in_image(image_response[0].shot.image, (width, height),
                                                   source_name, target_id)
            if bboxes:
                print(f'Found bounding box for {source_name}')
                return bboxes, source_name
            else:
                self.tag_not_located = True
                print(f'Failed to find bounding box for {source_name}')
        return [], None
    
    def old_detect_fiducial_in_image(self, image, dim, source_name, target_id):
        """Detect the fiducial within a single image and return its bounding box."""
        image_grey = np.array(
            Image.frombytes('P', (int(dim[0]), int(dim[1])), data=image.data, decoder_name='raw'))

        #Rotate each image such that it is upright
        image_grey = self.rotate_image(image_grey, source_name)

        #Make the image greyscale to use bounding box detections
        detector = apriltag(family='tag36h11')
        detections = detector.detect(image_grey)

        bboxes = []
        for i in range(len(detections)):
            # Draw the bounding box detection in the image.
            if (detections[i]["id"] != target_id): continue # Only add the bounding box of the target id's fiducial
            bbox = detections[i]['lb-rb-rt-lt']
            cv2.polylines(image_grey, [np.int32(bbox)], True, (0, 0, 0), 2)
            bboxes.append(bbox)

        self.image[source_name] = image_grey
        return bboxes

    def old_pixel_coords_to_camera_coords(self, bbox, intrinsics, source_name):
        """Compute transformation of 2d pixel coordinates to 3d camera coordinates."""
        camera = self.make_camera_matrix(intrinsics)
        # Track a triplet of (translation vector, rotation vector, camera source name)
        best_bbox = (None, None, source_name)
        # The best bounding box is considered the closest to the robot body.
        closest_dist = float('inf')
        for i in range(len(bbox)):
            obj_points, img_points = self.bbox_to_image_object_pts(bbox[i])
            if self.camera_to_extrinsics_guess[source_name][0]:
                # initialize the position estimate with the previous extrinsics solution
                # then iteratively solve for new position
                old_rvec, old_tvec = self.camera_to_extrinsics_guess[source_name][1]
                _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)),
                                             old_rvec, old_tvec, True, cv2.SOLVEPNP_ITERATIVE)
            else:
                # Determine current extrinsic solution for the tag.
                _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)))

            # Save extrinsics results to help speed up next attempts to locate bounding box in
            # the same camera source.
            self.camera_to_extrinsics_guess[source_name] = (True, (rvec, tvec))

            dist = math.sqrt(float(tvec[0][0])**2 + float(tvec[1][0])**2 +
                             float(tvec[2][0])**2) / 1000.0
            if dist < closest_dist:
                closest_dist = dist
                best_bbox = (tvec, rvec, source_name)

        # Flag indicating if the best april tag been found/located
        self.tag_not_located = best_bbox[0] is None and best_bbox[1] is None
        return best_bbox
    
    def old_compute_fiducial_in_world_frame(self, tvec):
        """Transform the tag position from camera coordinates to world coordinates."""
        fiducial_rt_camera_frame = np.array(
            [float(tvec[0][0]) / 1000.0,
             float(tvec[1][0]) / 1000.0,
             float(tvec[2][0]) / 1000.0])
        body_tform_fiducial = (self.camera_tform_body.inverse()).transform_point(
            fiducial_rt_camera_frame[0], fiducial_rt_camera_frame[1], fiducial_rt_camera_frame[2])
        fiducial_rt_world = self.body_tform_world.inverse().transform_point(
            body_tform_fiducial[0], body_tform_fiducial[1], body_tform_fiducial[2])
        return fiducial_rt_world

    def get_fiducial_rt_world_from_id(self, target_id):
        fiducial_rt_world = None

        # Detect the april tag in the images from Spot using the apriltag library.
        bboxes, source_name = self.old_image_to_bounding_box(target_id)
        if bboxes:
            self.previous_source = source_name
            (tvec, _, source_name) = self.old_pixel_coords_to_camera_coords(
                bboxes, self.intrinsics, source_name)
            vision_tform_fiducial_position = self.old_compute_fiducial_in_world_frame(tvec)
            fiducial_rt_world = geometry_pb2.Vec3(x=vision_tform_fiducial_position[0],
                                                    y=vision_tform_fiducial_position[1],
                                                    z=vision_tform_fiducial_position[2])
        

        return fiducial_rt_world
def grasp_object_callback(self, goal_handle):
    # Provide the position of the goal tag

    self.record_flag = True

    self.timings["grasp_start"] = time.time()

    msg = goal_handle.request.pose

    # self.get_logger().warn("______________________")
    # self.get_logger().warn("______________________")
    # self.get_logger().warn("______________________")
    # self.get_logger().warn(str(msg.transform.translation.x))
    # self.get_logger().warn(str(msg.transform.translation.y))
    # self.get_logger().warn(str(msg.transform.translation.z))

    self.object_name = msg.child_frame_id

    if self.stop_robot_flag:
        self.get_logger().error("ROBOT STOPPED - GRASP OBJECT STOPPED")

        result = GraspObject.Result()
        result.success = False
        return result

    try:
        self.vision_tform_reference
    except:
        self.get_logger().warn("SPOT NOT CALIBRATED")

        result = GraspObject.Result()
        result.success = False
        return result

    # Create transform from message
    reference_tform_object = SE3Pose(
        x=msg.transform.translation.x,
        y=msg.transform.translation.y,
        z=msg.transform.translation.z,
        rot=Quat(
            w=msg.transform.rotation.w,
            x=msg.transform.rotation.x,
            y=msg.transform.rotation.y,
            z=msg.transform.rotation.z,
        ),
    )
    # Look up euler angles
    axis = quat2euler(
        [
            msg.transform.rotation.w,
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
        ]
    )

    self.get_logger().info(f"AXIS: {axis}")

    def quadrant(num):
        if np.pi / 4 < num < 3 * np.pi / 4:
            return 4
        elif 0 < num < np.pi / 4 or -np.pi / 4 < num < 0:
            return 3
        elif -3 * np.pi / 4 < num < -np.pi / 4:
            return 2
        elif 3 * np.pi / 4 < num < np.pi or -np.pi < num < -3 * np.pi / 4:
            return 1
        else:
            return 5

    rotate = 0

    # Check if inverted orientation
    if self.object_name == "tupperware":
        axis0 = quadrant(axis[0])
        axis1 = quadrant(axis[1])
        axis2 = quadrant(axis[2])

        if axis0 == 4 and axis1 == 3 and axis2 == 3:
            rotate = 2
        elif axis0 == 2 and axis1 == 3 and axis2 == 1:
            rotate = 2
        elif axis0 == 2 and axis1 == 4 and axis2 == 1:
            rotate = 2

        elif axis0 == 4 and axis1 == 3 and axis2 == 4:
            rotate = 1
        elif axis0 == 2 and axis1 == 3 and axis2 == 4:
            rotate = 1
        elif axis0 == 1 and axis1 == 3 and axis2 == 4:
            rotate = 1

        elif axis0 == 4 and axis1 == 3 and axis2 == 1:
            rotate = 0
        elif axis0 == 2 and axis1 == 3 and axis2 == 3:
            rotate = 0
        elif axis0 == 2 and axis1 == 2 and axis2 == 3:
            rotate = 0

        elif axis0 == 4 and axis1 == 3 and axis2 == 2:
            rotate = 3
        elif axis0 == 2 and axis1 == 3 and axis2 == 2:
            rotate = 3
        elif axis0 == 3 and axis1 == 2 and axis2 == 2:
            rotate = 3
        elif axis0 == 1 and axis1 == 3 and axis2 == 2:
            rotate = 3
        elif axis0 == 3 and axis1 == 3 and axis2 == 2:
            rotate = 3

        else:
            self.get_logger().warn(str(axis0) + str(axis1) + str(axis2))

    self.get_logger().info(f"ROTATE? {rotate}")

    # Check if tupperware needs to be rotated
    if self.object_name == "tupperware" and rotate > 0:
        if rotate == 1:
            inverted_rotation = SE3Pose(0.0, 0.0, 0.0, Quat(0.707, 0.0, 0.707, 0.0))
        elif rotate == 2:
            inverted_rotation = SE3Pose(0.0, 0.0, 0.0, Quat(0.0, 0.0, 1.0, 0.0))
        elif rotate == 3:
            inverted_rotation = SE3Pose(0.0, 0.0, 0.0, Quat(0.707, 0.0, -0.707, 0.0))
        vision_tform_object = (
            self.vision_tform_reference * reference_tform_object * inverted_rotation
        )
    else:
        vision_tform_object = self.vision_tform_reference * reference_tform_object

    # Determine pre-grasp object type
    if self.object_name in self.object_list:
        self.get_the_grasps()
        object_tform_hand = SE3Pose(
            self.grasp_poses[self.object_name]["pre"][0],
            self.grasp_poses[self.object_name]["pre"][1],
            self.grasp_poses[self.object_name]["pre"][2],
            self.grasp_poses[self.object_name]["quat"],
        )
    else:
        # Object unknown
        object_tform_hand = SE3Pose(0.0, 0.0, 0.0, Quat(1.0, 0.0, 0.0, 0.0))

    vision_tform_hand = vision_tform_object * object_tform_hand

    arm_command = RobotCommandBuilder.arm_pose_command(
        x=vision_tform_hand.x,
        y=vision_tform_hand.y,
        z=vision_tform_hand.z,
        qw=vision_tform_hand.rot.w,
        qx=vision_tform_hand.rot.x,
        qy=vision_tform_hand.rot.y,
        qz=vision_tform_hand.rot.z,
        frame_name=VISION_FRAME_NAME,
        seconds=5,
    )
    arm_command.synchronized_command.arm_command.arm_cartesian_command.max_linear_velocity.value = self.max_arm_lin_vel
    arm_command.synchronized_command.arm_command.arm_cartesian_command.max_angular_velocity.value = self.max_grasp_ang_vel
    arm_command.synchronized_command.arm_command.arm_cartesian_command.maximum_acceleration.value = self.max_grasp_accel

    self.get_logger().info("COMMANDING ROBOT")
    m1 = self.SE3Pose_to_TransformStamped(vision_tform_object, "vision", "spot_object")
    self.get_logger().error(str(m1))
    self.get_logger().error("DESIRED HAND 1242")
    self.get_logger().error("DESIRED HAND 1242")
    self.get_logger().error("DESIRED HAND 1242")
    self.get_logger().error("DESIRED HAND 1242")
    self.get_logger().error("DESIRED HAND 1242")
    m2 = self.SE3Pose_to_TransformStamped(vision_tform_hand, "vision", "desired_hand")
    self.get_logger().error(str(m2))

    tfm = TFMessage()
    tfm.transforms = [m1, m2]

    self.tf_pub.publish(tfm)

    # Send arm command to robot
    if not self.stop_robot_flag:
        cmd_id = self.robot_command_client.robot_command(arm_command)
        self.get_logger().info("STARTING MOVEMENT")
        output = self.block_until_arm_arrives_interruptable(
            self.robot_command_client, cmd_id, timeout_sec=10
        )
    else:
        self.get_logger().error("ROBOT STOPPED - PREGRASP STOPPED")
        result = GraspObject.Result()
        return result

    # Open gripper
    if not self.stop_robot_flag:
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()
        cmd_id = self.robot_command_client.robot_command(gripper_command)
        output = self.block_until_gripper_arrives(
            self.robot_command_client, cmd_id, timeout_sec=2
        )

        self.get_logger().info("MOVED TO INITAL GRASP POSITION")

        time.sleep(1)
    else:
        self.get_logger().error("ROBOT STOPPED - PREGRASP OPEN STOPPED")
        result = GraspObject.Result()
        return result

    # Move to grasp position

    # Determine grasp object type
    if self.object_name in self.object_list:
        self.get_the_grasps()
        object_tform_hand = SE3Pose(
            self.grasp_poses[self.object_name]["pos"][0],
            self.grasp_poses[self.object_name]["pos"][1],
            self.grasp_poses[self.object_name]["pos"][2],
            self.grasp_poses[self.object_name]["quat"],
        )
    else:
        # Unknown object
        self.get_logger().error("No object detected for grasping!")

        result = GraspObject.Result()
        return result

    vision_tform_hand = vision_tform_object * object_tform_hand

    arm_command = RobotCommandBuilder.arm_pose_command(
        x=vision_tform_hand.x,
        y=vision_tform_hand.y,
        z=vision_tform_hand.z,
        qw=vision_tform_hand.rot.w,
        qx=vision_tform_hand.rot.x,
        qy=vision_tform_hand.rot.y,
        qz=vision_tform_hand.rot.z,
        frame_name=VISION_FRAME_NAME,
        seconds=5,
    )
    arm_command.synchronized_command.arm_command.arm_cartesian_command.max_linear_velocity.value = self.max_arm_lin_vel
    arm_command.synchronized_command.arm_command.arm_cartesian_command.max_angular_velocity.value = self.max_grasp_ang_vel
    arm_command.synchronized_command.arm_command.arm_cartesian_command.maximum_acceleration.value = self.max_grasp_accel

    # Send arm command to robot
    if not self.stop_robot_flag:
        cmd_id = self.robot_command_client.robot_command(arm_command)
        output = self.block_until_arm_arrives_interruptable(
            self.robot_command_client, cmd_id, timeout_sec=10
        )

        self.get_logger().info("MOVED TO GRASP LOCATION")
    else:
        self.get_logger().error("ROBOT STOPPED - GRASP MOVEMENT STOPPED")
        result = GraspObject.Result()
        return result

    if self.object_name in self.object_list:
        self.get_the_grasps()
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            self.grasp_poses[self.object_name]["gripper_close"]
        )
        gripper_command.synchronized_command.gripper_command.claw_gripper_command.maximum_open_close_velocity.value = self.max_gripper_vel
        if not self.stop_robot_flag:
            cmd_id = self.robot_command_client.robot_command(gripper_command)
            output = self.block_until_gripper_arrives(
                self.robot_command_client, cmd_id, timeout_sec=2
            )
        else:
            self.get_logger().error("ROBOT STOPPED - GRIPPING OBJECT STOPPED")
            output = False
            result = GraspObject.Result()
            return result
    else:
        self.get_logger().error("No object detected for grasping!")
        output = False
        result = GraspObject.Result()
        return result

    self.get_logger().info("GRASPED OBJECT")

    time.sleep(2)

    # check transform from tag to hand (if available)
    robot_state_transforms = (
        self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    )
    vision_tform_hand = get_a_tform_b(
        robot_state_transforms, VISION_FRAME_NAME, HAND_FRAME_NAME
    )

    ref_tform_vision = self.vision_tform_reference.inverse()
    ref_tform_hand = ref_tform_vision * vision_tform_hand

    if self.object_name == "bleach":
        self.get_the_grasps()
        self.grasp_poses["bleach"]["post"] = self.grasp_poses["bleach"]["pose_preset"]
    elif self.object_name is not None:
        if self.tf_buffer.can_transform(
            self.base_frame, self.object_name, rclpy.time.Time()
        ):
            msg = self.tf_buffer.lookup_transform(
                self.base_frame, self.object_name, rclpy.time.Time()
            )
            ref_tform_object = SE3Pose(
                x=msg.transform.translation.x,
                y=msg.transform.translation.y,
                z=msg.transform.translation.z,
                rot=Quat(
                    msg.transform.rotation.w,
                    msg.transform.rotation.x,
                    msg.transform.rotation.y,
                    msg.transform.rotation.z,
                ),
            )

            if self.object_name == "tupperware" and rotate > 0:
                ref_tform_object = ref_tform_object * inverted_rotation

            object_tform_hand = ref_tform_object.inverse() * ref_tform_hand
            self.get_logger().warn(str(object_tform_hand))
            self.get_logger().warn(str(self.object_name))
            if self.object_name in self.object_list:
                self.get_the_grasps()
                self.grasp_poses[self.object_name]["post"] = [
                    object_tform_hand.x,
                    object_tform_hand.y,
                    object_tform_hand.z,
                ]
                self.grasp_poses[self.object_name]["post_quat"] = Quat(
                    object_tform_hand.rot.w,
                    object_tform_hand.rot.x,
                    object_tform_hand.rot.y,
                    object_tform_hand.rot.z,
                )
                self.get_logger().warn(
                    "OBJECT GRASP TRANSFORM UPDATED: "
                    + str(self.grasp_poses[self.object_name]["post"])
                )

    # If knife, determine location to pull knife out of block after grasp
    if self.object_name in ["knife"]:
        self.get_the_grasps()
        object_tform_new_hand = SE3Pose(
            self.grasp_poses[self.object_name]["out"][0],
            self.grasp_poses[self.object_name]["out"][1],
            self.grasp_poses[self.object_name]["out"][2],
            self.grasp_poses[self.object_name]["post_quat"],
        )
    elif self.object_name in ["tupperware", "sponge", "bleach", "broom", "test"]:
        goal_handle.succeed()

        result = GraspObject.Result()
        result.success = output
        return result
    else:
        # Unknown object
        self.get_logger().error("No object detected for grasping!")

        result = GraspObject.Result()
        return result

    # only continue if knife
    vision_tform_hand = vision_tform_object * object_tform_new_hand

    arm_command = RobotCommandBuilder.arm_pose_command(
        x=vision_tform_hand.x,
        y=vision_tform_hand.y,
        z=vision_tform_hand.z,
        qw=vision_tform_hand.rot.w,
        qx=vision_tform_hand.rot.x,
        qy=vision_tform_hand.rot.y,
        qz=vision_tform_hand.rot.z,
        frame_name=VISION_FRAME_NAME,
        seconds=5,
    )
    arm_command.synchronized_command.arm_command.arm_cartesian_command.max_linear_velocity.value = self.max_arm_lin_vel
    arm_command.synchronized_command.arm_command.arm_cartesian_command.max_angular_velocity.value = self.max_grasp_ang_vel
    arm_command.synchronized_command.arm_command.arm_cartesian_command.maximum_acceleration.value = self.max_grasp_accel

    # Send arm command to robot
    if not self.stop_robot_flag:
        cmd_id = self.robot_command_client.robot_command(arm_command)
        output = self.block_until_arm_arrives_interruptable(
            self.robot_command_client, cmd_id, timeout_sec=10
        )
    else:
        self.get_logger().error("ROBOT STOPPED - POST GRASP MOVEMENT STOPPED")
        result = GraspObject.Result()
        return result

    time.sleep(2)

    robot_state_transforms = (
        self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    )
    vision_tform_hand = get_a_tform_b(
        robot_state_transforms, VISION_FRAME_NAME, HAND_FRAME_NAME
    )

    ref_tform_vision = self.vision_tform_reference.inverse()
    ref_tform_hand = ref_tform_vision * vision_tform_hand

    if self.object_name is not None:
        if self.tf_buffer.can_transform(
            self.base_frame, self.object_name, rclpy.time.Time()
        ):
            msg = self.tf_buffer.lookup_transform(
                self.base_frame, self.object_name, rclpy.time.Time()
            )
            ref_tform_object = SE3Pose(
                x=msg.transform.translation.x,
                y=msg.transform.translation.y,
                z=msg.transform.translation.z,
                rot=Quat(
                    msg.transform.rotation.w,
                    msg.transform.rotation.x,
                    msg.transform.rotation.y,
                    msg.transform.rotation.z,
                ),
            )

            object_tform_hand = ref_tform_object.inverse() * ref_tform_hand
            self.get_logger().warn(str(object_tform_hand))
            self.get_logger().warn(str(self.object_name))
            if self.object_name in self.object_list:
                self.get_the_grasps()
                # for some reason, z doesn't update well in this position (probably due to camera angles, so keep old z)
                self.grasp_poses[self.object_name]["post"] = [
                    object_tform_hand.x,
                    object_tform_hand.y,
                    self.grasp_poses[self.object_name]["post"][2],
                ]
                self.grasp_poses[self.object_name]["post_quat"] = Quat(
                    object_tform_hand.rot.w,
                    object_tform_hand.rot.x,
                    object_tform_hand.rot.y,
                    object_tform_hand.rot.z,
                )
                self.get_logger().warn(
                    "OBJECT GRASP TRANSFORM UPDATED: "
                    + str(self.grasp_poses[self.object_name]["post"])
                )

    goal_handle.succeed()

    result = GraspObject.Result()
    result.success = output

    return result



# Main Function
def main():
    # Initalize ROS
    rclpy.init()

    # QApplication(sys.argv)

    robot_controller = RobotController()

    executor = MultiThreadedExecutor()
    executor.add_node(robot_controller)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Keyboard Interrupt called!")
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
