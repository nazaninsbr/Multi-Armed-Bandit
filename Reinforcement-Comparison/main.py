import numpy as np
import pandas as pd
import math
import random
from scipy.stats import beta

def create_point_cost_arrays(reward_df):
    action_reward_dists = {i-1: [] for i in range(1, 7)}
    for _, row in reward_df.iterrows():
        for i in range(1, 7):
            action_reward_dists[i-1].append([row['Product '+str(i)], row['cost '+str(i)]])
    return action_reward_dists

class ReinforcementComparison:
    def __init__(self, action_reward_dists, n_actions, alpha, beta, a):
        # first element of each list is the product, the second one is cost
        self.action_reward_dists = action_reward_dists
        self.n_actions = n_actions
        self.a = a
        self.alpha = alpha
        self.beta = beta
        self.temprature = 1
        # internal parameters
        self.R = np.zeros(n_actions)
        self.preference_for_action = np.zeros(n_actions)
        self.policy = np.zeros(n_actions)
        # plotting vars
        self.theoretical_y = [0]
        self.number_of_times_actions_were_played = np.zeros(n_actions)

    def get_reward_of_action(self, action):
        x = random.choice(self.action_reward_dists[action])
        r = self.a * x[0] - x[1]
        return r

    def update_policy(self):
        sigma_value = 0
        for a in range(self.n_actions):
            sigma_value += np.exp(self.preference_for_action[a]/self.temprature)

        for ar in range(self.n_actions):
            self.policy[ar] = np.exp(self.preference_for_action[ar]/self.temprature)/sigma_value

    def select_action(self):
        random_value = np.random.random()
        if random_value < self.policy[0]:
            return 0
        if random_value < self.policy[0]+self.policy[1]:
            return 1
        if random_value < self.policy[0]+self.policy[1]+self.policy[2]:
            return 2
        if random_value < self.policy[0]+self.policy[1]+self.policy[2]+self.policy[3]:
            return 3
        if random_value < self.policy[0]+self.policy[1]+self.policy[2]+self.policy[3]+self.policy[4]:
            return 4
        return 5

    def update_preference_for_action(self, action, reward):
        self.preference_for_action[action] = self.preference_for_action[action] + self.beta * (reward - self.R[action])

    def normalize_preferences(self):
        max_value = max([self.preference_for_action[action] for action in range(self.n_actions)])
        min_value = min([self.preference_for_action[action] for action in range(self.n_actions)])

        for action in range(self.n_actions):
            self.preference_for_action[action] = (self.preference_for_action[action]-min_value)/(max_value-min_value)

    def update_value(self, action, reward):
        self.R[action] = self.R[action] + self.alpha * (reward - self.R[action])

    def train(self, number_of_iterations):
        for i in range(1, number_of_iterations+1):
            self.update_policy()

            action = self.select_action()
            self.number_of_times_actions_were_played[action] += 1
            reward = self.get_reward_of_action(action)

            self.update_preference_for_action(action, reward)
            self.normalize_preferences()
            self.update_value(action, reward)
            
        print('Number of times each action was tried:', self.number_of_times_actions_were_played)
        print('Preference:', self.preference_for_action)
        print('Policy:', self.policy)
        print('Values:', self.R)

def main():
    reward_df = pd.read_csv('Dataset.csv')
    action_reward_dists = create_point_cost_arrays(reward_df)

    print('Preference Normalization method:')
    alpha, beta = 0.1, 0.9
    a = 2.5
    rcl = ReinforcementComparison(action_reward_dists, len(action_reward_dists.keys()), alpha, beta, a)
    rcl.train(20000)

if __name__ == '__main__':
    main()