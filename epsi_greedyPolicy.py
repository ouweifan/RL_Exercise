import numpy as np
import random

class epsi_greedyPolicy:
    def __init__(self, epsilon, action_set, initial_est):
        self.epsilon = epsilon
        self.estimate = np.full((len(action_set)), float(initial_est))
        self.action_stats = np.full((len(action_set)), 0)

    def getAction(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.action_stats)-1)
        else:
            return self.greedyAction()

    def greedyAction(self):
        curr_act = [0]
        curr_val = self.estimate[0]
        for i in range(1, len(self.estimate)):
            if self.estimate[i] > curr_val:
                curr_val = self.estimate[i]
                curr_act = [i]
            elif self.estimate[i] == curr_val:
                curr_act.append(i)
        return random.choice(curr_act)

    def updateEstimate(self, action, reward):
        self.action_stats[action] += 1
        self.estimate[action] += (1/float(self.action_stats[action])) * (reward - self.estimate[action])