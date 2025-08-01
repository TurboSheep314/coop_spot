#!/usr/bin/env python3

import numpy as np
# import pyqtgraph as pg
# from PyQt5 import QtWidgets


# Need a way to ensure this does not terminate on round 1#
# --- PARAMETERS ---
NUM_ROUNDS = 10
true_probs = [1, 0]  # True probabilities for each arm
temperature_boltzmann = 3  # temp for Boltzmann policy, maybe look at if this is predicted with a survey somwhere
num_arms = len(true_probs)

# --- TRACKING ---
reward_history = {
    "Boltzmann": [],
    "UCB": [],  # , commented to run reward history
    # "random": [],
    # "Participant":[] commented because it is not used on the robot, only for simulation here
}
# creates the arrays for the policies
arm_counts_ucb = np.zeros(num_arms)
arm_rewards_ucb = np.zeros(num_arms)

arm_counts_boltz = np.zeros(num_arms)
arm_rewards_boltz = np.zeros(num_arms)


# --- POLICIES ---


# this is the part that allows user input
def Person():
    action = input("Enter your action (S or W): ")
    if action.upper() == "S":
        reward = 0
    elif action.upper() == "W":
        reward = 1
    else:
        print("Invalid input. Please enter 'S' or 'W'.")
    return reward


# This is the simulated human strategy
def boltzmann(ev, temperature):
    ev = np.array(ev)
    scaled_ev = ev / temperature
    scaled_ev -= np.max(scaled_ev)
    exp_ev = np.exp(scaled_ev)
    probs = exp_ev / np.sum(exp_ev)
    return np.random.choice(len(ev), p=probs)


# this is the robot poicy
def ucb1(t, counts, rewards):
    ucb_vals = np.zeros(num_arms)
    for i in range(num_arms):
        if counts[i] == 0:
            return i  # Prevents a divide by zero
        avg = rewards[i] / counts[i]
        bound = np.sqrt(2 * np.log(t + 1) / counts[i])
        ucb_vals[i] = avg + bound
    return np.argmax(ucb_vals)


# --- SIMULATION LOOP ---
def simulation_loop(num_rounds):
    for t in range(num_rounds):
        # --- BOLTZMANN ---
        est_vals_boltz = np.divide(
            arm_rewards_boltz,
            arm_counts_boltz,
            out=np.zeros(num_arms),
            where=arm_counts_boltz > 0,
        )
        a_boltz = boltzmann(est_vals_boltz, temperature_boltzmann)
        r_boltz = 1 if np.random.rand() < true_probs[a_boltz] else 0
        arm_counts_boltz[a_boltz] += 1
        arm_rewards_boltz[a_boltz] += r_boltz
        reward_history["Boltzmann"].append(r_boltz)

        # --- UCB1 ---
        a_ucb = ucb1(t, arm_counts_ucb, arm_rewards_ucb)
        r_ucb = 1 if np.random.rand() < true_probs[a_ucb] else 0
        arm_counts_ucb[a_ucb] += 1
        arm_rewards_ucb[a_ucb] += r_ucb
        reward_history["UCB"].append(r_ucb)
    return reward_history


# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()

#         # Plot
#         self.plot_graph = pg.PlotWidget()
#         self.setCentralWidget(self.plot_graph)
#         self.plot_graph.setBackground("k")
#         pen = pg.mkPen(color=(255, 0, 0))
#         self.plot_graph.setTitle(
#             "Bandit Strategies: Boltzmann vs UCB1", color="w", size="18pt"
#         )
#         styles = {"color": "white", "font-size": "18px"}
#         self.plot_graph.setLabel("left", "Cumulative Reward", **styles)
#         self.plot_graph.setLabel("bottom", "Round", **styles)
#         self.plot_graph.addLegend()
#         self.plot_graph.showGrid(x=True, y=True)
#         # self.plot_graph.setYRange(20, 40)
#         self.Round = np.arange(0, NUM_ROUNDS)
#         self.cum_reward = {}
#         self.line = {}
#         for label in reward_history:
#             self.cum_reward[label] = np.cumsum(reward_history[label])
#             self.line[label] = self.plot_graph.plot(
#                 self.Round,
#                 self.cum_reward[label],
#                 name=label.upper(),
#                 pen=pen,
#                 symbol="o",
#                 symbolSize=15,
#                 symbolBrush="b",
#             )

#     # Add a timer to simulate new temperature measurements
#     self.timer = QtCore.QTimer()
#     self.timer.setInterval(300)
#     self.timer.timeout.connect(self.update_plot)
#     self.timer.start()

# def update_plot(self):
#     self.time = self.time[1:]
#     self.time.append(self.time[-1] + 1)
#     self.temperature = self.temperature[1:]
#     self.temperature.append(randint(20, 40))
#     self.line.setData(self.time, self.temperature)


# app = QtWidgets.QApplication([])
# main = MainWindow()
# main.show()
# app.exec()
