import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def read_csv(filename):
    return pd.read_csv(filename)

class UCB1Learning:
    def __init__(self, n_actions, dist_df):
        self.number_of_times_action_was_choosen = np.zeros(n_actions)
        self.dist_df = dist_df
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.t = 0
        self.ut = np.zeros(n_actions)
        
    def get_reward_of_ad(self, this_action):
        this_actions_distribution = self.dist_df.iloc[:, this_action]
        this_actions_reward_number = np.random.randint(len(this_actions_distribution)) 
        return this_actions_distribution[this_actions_reward_number]

    def update_value(self, action, reward):
        lr = 1/self.number_of_times_action_was_choosen[action]
        self.Q[action] = self.Q[action]+lr*(reward - self.Q[action])
        
    def estimate_value(self, action, reward): 
        self.Q[action] = reward 

    def run_all_actions_once(self):
        for i in range(self.n_actions):
            self.t = self.t + 1
            reward = self.get_reward_of_ad(i)
            
            self.estimate_value(i, reward)
            self.number_of_times_action_was_choosen[i] += 1
            
    def select_action(self):
        return np.argmax(self.ut) 
    
    def update_ut(self, i):
        self.ut[i] = self.Q[i] + math.sqrt(2*np.log(self.t)/self.number_of_times_action_was_choosen[i])
    
    def train(self, num_iterations):
        for _ in range(num_iterations):
            for i in range(self.n_actions):
                self.update_ut(i)
                
            action = self.select_action()
            reward = self.get_reward_of_ad(action)
            
            self.t = self.t + 1
            self.number_of_times_action_was_choosen[action] = self.number_of_times_action_was_choosen[action] + 1
            
            self.update_value(action, reward)
        
        print("Q =", self.Q)
        print("Best Ad =", np.argmax(self.Q)+1)

def main():
    df = read_csv('./Ads_Optimisation.csv')
    number_of_actions = len(df.columns)

    ucb1 = UCB1Learning(number_of_actions, df)
    ucb1.run_all_actions_once()
    for trial_number in [100, 1000, 10000, 100000]:
        print("Number of trials:", trial_number)
        ucb1.train(trial_number)
        print('------------------- (:) -------------------')

if __name__ == '__main__':
    main()