import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Vector3
from std_msgs.msg import String
import numpy as np
import math


class HandIntentEstimator(Node):
    def __init__(self):
        super().__init__("hand_intent_estimator")

        # Subscribers
        self.create_subscription(
            PointStamped, "/hand/keypoints", self.hand_keypoint_callback, 10
        )  # need to add to camera controller
        self.create_subscription(
            PointStamped, "/object_positions", self.object_callback, 10
        )  # Need to add to camera controller

        # Publishers
        self.intent_pub = self.create_publisher(String, "/hand_intent/target", 10)
        self.orientation_pub = self.create_publisher(Vector3, "/hand/orientation", 10)
        self.velocity_pub = self.create_publisher(Vector3, "/hand/velocity", 10)

        self.prev_hand_pos = None
        self.object_positions = {}  # Dictionary of object_id: PointStamped
        self.hand_keypoints = {}  # {id: PointStamped}
        self.prev_posterior = None
        self.beta_bins = [
            10,
            30,
            60,
            90,
            120,
            150,
            180,
        ]  # example bin edges.  These are the degrees that represent the liklehood of the target.  0-30 = good, 150-180 = bad
        self.gamma_thresholds = {"flexion": 45, "neutral": 120}

        # CPT structure: P(beta_bin, gamma_class | object_id) this will need to be trained or refined based on the experiment
        self.cpt = {
            "banana": {
                (0, "flexion"): 0.9,
                (1, "neutral"): 0.05,
                (2, "other"): 0.05,
            },
            "milk": {
                (0, "neutral"): 0.8,
                (1, "flexion"): 0.1,
                (2, "other"): 0.1,
            },
            "cereal": {
                (0, "flexion"): 0.7,
                (1, "neutral"): 0.2,
                (2, "other"): 0.1,
            },
        }

    def get_beta_bin(beta, bin_edges):
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= beta < bin_edges(i + 1):
                return i
        return len(bin_edges) - 2  # this takes care of the beta==180

    def hand_keypoint_callback(self, msg):
        self.hand_keypoints[msg.header.frame_id] = msg

        if "WRIST" in self.hand_keypoints:
            wrist = self.hand_keypoints["WRIST"].point
            if self.prev_hand_pos:
                dt = 1.0 / 15.0  # assuming 15 FPS
                vx = (wrist.x - self.prev_hand_pos.x) / dt
                vy = (wrist.y - self.prev_hand_pos.y) / dt
                vz = (wrist.z - self.prev_hand_pos.z) / dt
                self.velocity_pub.publish(Vector3(x=vx, y=vy, z=vz))
            self.prev_hand_pos = wrist

        if all(k in self.hand_keypoints for k in ["WRIST", "INDEX_FINGER", "PINKY"]):
            self.estimate_orientation()
            self.estimate_intent()

    def object_callback(self, msg):
        self.object_positions[msg.header.frame_id] = msg.point

    def estimate_orientation(self):
        wrist = np.array(
            [
                self.hand_keypoints["WRIST"].point.x,
                self.hand_keypoints["WRIST"].point.y,
                self.hand_keypoints["WRIST"].point.z,
            ]
        )
        index = np.array(
            [
                self.hand_keypoints["INDEX_FINGER"].point.x,
                self.hand_keypoints["INDEX_FINGER"].point.y,
                self.hand_keypoints["INDEX_FINGER"].point.z,
            ]
        )
        pinky = np.array(
            [
                self.hand_keypoints["PINKY"].point.x,
                self.hand_keypoints["PINKY"].point.y,
                self.hand_keypoints["PINKY"].point.z,
            ]
        )

        v1 = index - wrist
        v2 = pinky - wrist
        normal = np.cross(v1, v2)
        z_axis = np.array([0, 0, 1])
        gamma = math.degrees(
            math.acos(
                np.dot(normal, z_axis)
                / (np.linalg.norm(normal) * np.linalg.norm(z_axis))
            )
        )

        if gamma < self.gamma_thresholds["flexion"]:
            self.hand_pose = "flexion"
        elif gamma < self.gamma_thresholds["neutral"]:
            self.hand_pose = "neutral"
        else:
            self.hand_pose = "other"

        self.orientation_pub.publish(Vector3(x=normal[0], y=normal[1], z=normal[2]))

    def estimate_intent(self):
        if self.prev_hand_pos is None:
            return

        wrist = np.array(
            [self.prev_hand_pos.x, self.prev_hand_pos.y, self.prev_hand_pos.z]
        )
        velocity = np.array(
            [self.prev_hand_pos.x, self.prev_hand_pos.y, self.prev_hand_pos.z]
        )  # already computed

        posteriors = {}
        evidence_probs = []

        for obj_id, pos in self.object_positions.items():
            obj_vec = np.array([pos.x, pos.y, pos.z]) - wrist
            beta = math.degrees(
                math.acos(
                    np.dot(velocity, obj_vec)
                    / (np.linalg.norm(velocity) * np.linalg.norm(obj_vec))
                )
            )
            beta_bin = min(
                range(len(self.beta_bins)), key=lambda i: abs(beta - self.beta_bins[i])
            )

            likelihood = self.cpt.get(obj_id, {}).get((beta_bin, self.hand_pose), 1e-5)
            prior = (
                self.prev_posterior.get(obj_id, 1.0 / len(self.object_positions))
                if self.prev_posterior
                else 1.0 / len(self.object_positions)
            )

            posterior = likelihood * prior
            posteriors[obj_id] = posterior
            evidence_probs.append(posterior)

        norm_factor = sum(evidence_probs)
        for obj in posteriors:
            posteriors[obj] /= norm_factor

        self.prev_posterior = posteriors
        best_obj = max(posteriors.items(), key=lambda x: x[1])[0]
        self.intent_pub.publish(String(data=best_obj))


def main(args=None):
    rclpy.init(args=args)
    node = HandIntentEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
