import numpy as np
import random
import matplotlib.pyplot as plt 

from bandit import bandit
from epsi_greedyPolicy import epsi_greedyPolicy


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
    # Done learning

    plt.subplot(211)
    plt.title("Average reward")
    plt.plot(chartx, avgValues[0],'r-', label="epsilon = 0.2")
    plt.plot(chartx, avgValues[1],'b-', label="epsilon = 0.1")
    plt.plot(chartx, avgValues[2],'g-', label="epsilon = 0.01")
    plt.plot(chartx, avgValues[3],'k--', label="greedy")
    plt.xlim((0, NUM_CYCLES))
    plt.xlabel("Time Step")


    plt.subplot(212)
    plt.title("%Optimal Action")
    plt.plot(chartx, optimalActionCnt[0],'r-', label="epsilon = 0.2")
    plt.plot(chartx, optimalActionCnt[1],'b-', label="epsilon = 0.1")
    plt.plot(chartx, optimalActionCnt[2],'g-', label="epsilon = 0.01")
    plt.plot(chartx, optimalActionCnt[3],'k--', label="greedy")
    plt.xlim((0, NUM_CYCLES))
    plt.ylim((-0.01, 1))
    plt.xlabel("Time Step")
    plt.legend(bbox_to_anchor=(0,-0.3,0,0), loc='lower left', ncol=4, borderaxespad=0.)
    
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()