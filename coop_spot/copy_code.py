def spot_arm_calibrate_pos(self):
    # this moves Spot's arm into the calibrate position

    # ensures Spot's gripper is closed
    message = Gripper.Goal()
    message.gripper_state = True
    self.gripper_service_client.wait_for_server()
    output = self.gripper_service_client.send_goal_async(message)

    # Create arm pose message
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
    output = self.desired_arm_body_pose_nc_service_client.send_goal_async(msg)


def spot_arm_deploy_calibrate(self):
    # this moves Spot's arm into the calibrate position

    # ensures Spot's gripper is closed
    message = Gripper.Goal()
    message.gripper_state = True
    self.gripper_service_client.wait_for_server()
    output = self.gripper_service_client.send_goal_async(message)

    # Create arm pose message
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
    self.desired_arm_body_pose_service_client.wait_for_server()
    return self.desired_arm_body_pose_service_client.send_goal(msg)


####grasp object


def spot_arm_grasp_object(self):
    msg = None

    self.get_logger().info("TRYING TO GRASP:" + str(self.seen_object))

    for i in range(5):
        if self.seen_object is not None:
            if self.tf_buffer.can_transform(
                self.base_frame, self.seen_object, rclpy.time.Time()
            ):
                msg = self.tf_buffer.lookup_transform(
                    self.base_frame, self.seen_object, rclpy.time.Time()
                )
                self.get_logger().info(f"FOUND {self.seen_object} at {msg}")
                break
        else:
            self.get_logger().error("seen_object is None")
    if msg is not None:
        message = GraspObject.Goal()
        message.pose = msg
        self.grasp_object_service_client.wait_for_server()
        return self.grasp_object_service_client.send_goal(message)
    else:
        self.get_logger().info("CAN'T DETECT OBJECT")
