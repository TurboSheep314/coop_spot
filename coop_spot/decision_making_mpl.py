#!/usr/bin/env python3

import numpy as np
from PyQt5 import QtWidgets
import sys
import matplotlib
import mplcyberpunk  # noqa: F401
import matplotlib as mpl

matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

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
for t in range(NUM_ROUNDS):
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


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        with mpl.style.context("cyberpunk"):
            sc = MplCanvas(self, width=10, height=6, dpi=100)

            # --- PLOT RESULTS ---
            for label in reward_history:
                sc.axes.plot(np.cumsum(reward_history[label]), label=label.capitalize())

                sc.axes.set_xlabel("Round", fontsize=15)
                sc.axes.set_ylabel("Cumulative Reward", fontsize=15)
                sc.axes.set_title("Bandit Strategies: Boltzmann vs UCB1", fontsize=20)
                sc.axes.legend(loc="best", fontsize=15)
                sc.axes.grid(True)
                # sc.axes.tight_layout()

            self.setCentralWidget(sc)
            self.show()


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
