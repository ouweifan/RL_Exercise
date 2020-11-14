import numpy as np
import random
import matplotlib.pyplot as plt 

class bandit:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
    
    def hit(self):
        return np.random.normal(self.mean, self.var)
    
class greedyPolicy:
    def __init__(self, action_set, initial_est):
        self.estimate = np.full((len(action_set)), float(initial_est))
        self.action_stats = np.full((len(action_set)), 0)
    def getAction(self):
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
        #print("esitmate before: ", self.estimate[action])
        self.action_stats[action] += 1
        self.estimate[action] += (1/float(self.action_stats[action])) * (reward - self.estimate[action])
        #print("action: ", action, " reward: ", reward, self.action_stats[action], self.estimate[action])

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

if __name__ == "__main__":
    k = 10              # Number of bandits
    NUM_CYCLES = 2000    # Number of cycles

    # Initialize bandits
    bandits = []
    means = np.random.randint(10, high=20, size=k)
    best_bandit = [0]
    best_bandit_val = means[0]
    for i in range(k):
        bandits.append(bandit(means[i], 3))
        if means[i] > best_bandit_val:
            best_bandit = [i]
            best_bandit_val = means[i]
        elif means[i] == best_bandit_val:
            best_bandit.append(i)

    policies = [
        epsi_greedyPolicy(0.2, range(k), 15),
        epsi_greedyPolicy(0.1, range(k), 15),
        epsi_greedyPolicy(0.01, range(k), 15),
        epsi_greedyPolicy(0.0, range(k), 15)
    ]
    num_policies = len(policies)

    chartx = range(1, NUM_CYCLES+1)
    avgValues = np.zeros((num_policies, NUM_CYCLES))
    optimalActionCnt = np.zeros((num_policies, NUM_CYCLES), dtype=float)
    action_history = np.zeros(NUM_CYCLES)
    optimalCnt = np.zeros(num_policies)
    totalReward = np.zeros(num_policies)

    # Start learning
    for i in range(1, NUM_CYCLES + 1):
        for j in range(num_policies):
            action = policies[j].getAction()
            reward = bandits[action].hit()
            policies[j].updateEstimate(action, reward)

            # Data logging
            totalReward[j] += reward
            if action in best_bandit:
                optimalCnt[j] += 1
            optimalActionCnt[j][i-1] = optimalCnt[j]/i
            avgValues[j][i-1] = totalReward[j]/i
            action_history[i-1] = action

    plt.subplot(211)
    plt.title("Average reward")
    plt.plot(chartx, avgValues[0],'r-', label="epsilon = 0.2")
    plt.plot(chartx, avgValues[1],'b-', label="epsilon = 0.1")
    plt.plot(chartx, avgValues[2],'g-', label="epsilon = 0.01")
    plt.plot(chartx, avgValues[3],'b--', label="greedy")
    plt.legend(bbox_to_anchor=(0., 0.02, 0.3, 1), loc='lower left', ncol=1, borderaxespad=0.)
    plt.xlim((0, NUM_CYCLES))
    plt.xlabel("Time Step")


    plt.subplot(212)
    plt.title("%Optimal Action")
    plt.plot(chartx, optimalActionCnt[0],'r-', label="epsilon = 0.2")
    plt.plot(chartx, optimalActionCnt[1],'b-', label="epsilon = 0.1")
    plt.plot(chartx, optimalActionCnt[2],'g-', label="epsilon = 0.01")
    plt.plot(chartx, optimalActionCnt[3],'b--', label="greedy")
    plt.xlim((0, NUM_CYCLES))
    plt.ylim((-0.01, 1))
    plt.xlabel("Time Step")
    
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()