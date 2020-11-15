import numpy as np
import random

class ucbPolicy:
    def __init__(self, c, action_set, initial_est):
        self.c = c
        self.estimate = np.full((len(action_set)), float(initial_est))
        self.action_stats = np.full((len(action_set)), 0)
        self.totalAction = 0

    def getAction(self):
        curr_act = [0]
        curr_ucb_val = self.getUcbVal(0)

        for i in range(1, len(self.estimate)):
            ucbOfI = self.getUcbVal(i)
            if ucbOfI > curr_ucb_val:
                curr_ucb_val = ucbOfI
                curr_act = [i]
            elif ucbOfI == curr_ucb_val:
                curr_act.append(i)
        self.totalAction += 1
        return random.choice(curr_act)
    
    def getUcbVal(self, action):
        if self.totalAction == 0:
            return float('-inf')
        elif self.action_stats[action] == 0:
            return float('inf')
        
        return self.estimate[action] + self.c * np.sqrt(np.log(self.totalAction)/self.action_stats[action])

    def updateEstimate(self, action, reward):
        self.action_stats[action] += 1
        self.estimate[action] += (1/float(self.action_stats[action])) * (reward - self.estimate[action])