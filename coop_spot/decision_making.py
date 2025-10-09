#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray


class DecisionMaker(Node):
    def __init__(self):
        super().__init__("decision_node")
        self.get_logger().info("Decision-making node started.")

        # --- ROS SUBSCRIPTIONS ---
        self.human_sub = self.create_subscription(
            Bool, "human_reward", self.person_callback, 10
        )
        self.robot_sub = self.create_subscription(
            Bool, "robot_reward", self.robot_callback, 10
        )

        # Publishers
        self.boltzmann_pub = self.create_publisher(
            Float64MultiArray, "/reward_history_boltzmann", 10
        )
        self.ucb_pub = self.create_publisher(
            Float64MultiArray, "/reward_history_ucb", 10
        )
        self.human_pub = self.create_publisher(
            Float64MultiArray, "/reward_history_human", 10
        )
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.reward_history_callback)

        # --- PARAMETERS ---So it seems you did a lot of progress, what's missing yet?
        self.num_arms = 2
        self.true_probs = [1, 0]  # simulated true arm probabilities
        self.temperature_boltzmann = 3.0

        # --- TRACKING ---
        self.round = 0
        self.human_cum_reward = 0
        self.robot_cum_reward = 0

        # UCB
        self.arm_counts_ucb = np.zeros(self.num_arms)
        self.arm_rewards_ucb = np.zeros(self.num_arms)

        # Boltzmann
        self.arm_counts_boltz = np.zeros(self.num_arms)
        self.arm_rewards_boltz = np.zeros(self.num_arms)

        self.reward_history = {
            "Boltzmann": [],
            "UCB": [],
            "Human": [],
        }

    def reward_history_callback(self):
        for key, data in self.reward_history.items():
            if key.lower() == "boltzmann":
                self.boltzmann_pub.publish(Float64MultiArray(data=data))
            elif key.lower() == "ucb":
                self.ucb_pub.publish(Float64MultiArray(data=data))
            elif key.lower() == "human":
                self.human_pub.publish(Float64MultiArray(data=data))

    # --- CALLBACKS ---

    def person_callback(self, msg: Bool):
        """Handles incoming human reward messages."""
        reward = 1 if msg.data else 0
        self.human_cum_reward += reward
        self.reward_history["Human"].append(reward)
        self.get_logger().info(
            f"[Human] Received reward: {reward}, Total: {self.human_cum_reward}"
        )

    def robot_callback(self, msg: Bool):
        """Handles incoming robot action and triggers UCB/Boltzmann logic."""
        reward = 1 if msg.data else 0
        self.robot_cum_reward += reward
        self.round += 1

        # --- UCB Policy ---
        arm_ucb = self.ucb1(self.round)
        #self.get_logger().info(f"bick bottle number {arm_ucb}")
        r_ucb = 1 if np.random.rand() < self.true_probs[arm_ucb] else 0
        self.arm_counts_ucb[arm_ucb] += 1
        self.arm_rewards_ucb[arm_ucb] += r_ucb
        self.reward_history["UCB"].append(r_ucb)

        # --- Boltzmann Policy ---
        est_vals = np.divide(
            self.arm_rewards_boltz,
            self.arm_counts_boltz,
            out=np.zeros_like(self.arm_rewards_boltz),
            where=self.arm_counts_boltz > 0,
        )
        arm_boltz = self.boltzmann(est_vals, self.temperature_boltzmann)
        r_boltz = 1 if np.random.rand() < self.true_probs[arm_boltz] else 0
        self.arm_counts_boltz[arm_boltz] += 1
        self.arm_rewards_boltz[arm_boltz] += r_boltz
        self.reward_history["Boltzmann"].append(r_boltz)

    # --- POLICIES ---

    def ucb1(self, t):
        ucb_vals = np.zeros(self.num_arms)
        for i in range(self.num_arms):
            if self.arm_counts_ucb[i] == 0:
                return i  # Explore untried arms
            avg = self.arm_rewards_ucb[i] / self.arm_counts_ucb[i]
            bound = np.sqrt(2 * np.log(t + 1) / self.arm_counts_ucb[i])
            ucb_vals[i] = avg + bound
        return int(np.argmax(ucb_vals))

    def boltzmann(self, est_values, temperature):
        if np.sum(est_values) == 0:
            return np.random.choice(self.num_arms)  # Initial exploration
        scaled_ev = est_values / temperature
        scaled_ev -= np.max(scaled_ev)  # for numerical stability
        exp_ev = np.exp(scaled_ev)
        probs = exp_ev / np.sum(exp_ev)
        return int(np.random.choice(len(est_values), p=probs))


# --- MAIN ---
def main(args=None):
    rclpy.init(args=args)
    node = DecisionMaker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
