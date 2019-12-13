import numpy as np
import matplotlib.pyplot as plt
import math

class SoftmaxLearning:
    def __init__(self, n_trials, n_actions, temperature):
        self.policy = np.zeros(n_actions)
        self.number_of_times_arm_was_choosen = np.zeros(n_actions) 
        self.arm_totals = np.zeros(n_actions) 
        self.Q = np.zeros(n_actions)
        self.n_actions = n_actions
        self.n_trials = n_trials 
        self.temperature = temperature
        
    def generate_reward(self, arm_number):
        if arm_number == 0:
            return np.random.normal(1, 0.5, 1)[0]
        if arm_number == 1:
            return np.random.normal(0, 1, 1)[0]
        if arm_number == 2:
            return np.random.normal(-2, 10, 1)[0]
        
    def pick_an_arm(self):
        random_value = np.random.random()
        if random_value < self.policy[0]:
            return 0
        if random_value < self.policy[0]+self.policy[1]:
            return 1
        return 2
        
    def get_reward(self, arm_number):
        return self.generate_reward(arm_number)
    
    def update_policy(self):
        sigma_value = 0
        for ar in range(self.n_actions):
            sigma_value += math.exp(self.Q[ar]/self.temperature)
            
        for ar in range(self.n_actions):
            self.policy[ar] = math.exp(self.Q[ar]/self.temperature)/sigma_value
    
    def update_value(self, choosen_arm, reward):
        lr = 1/self.number_of_times_arm_was_choosen[choosen_arm]
        self.Q[choosen_arm] = self.Q[choosen_arm]+lr*(reward - self.Q[choosen_arm])
        
    def plot(self, plot_data):
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['total'], label='total')
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['1'], label='arm 1')
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['2'], label='arm 2')
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['3'], label='arm 3')
        plt.legend()
        plt.xlabel('trial number')
        plt.show()
        
    def update_average_reward_values(self, plot_data):
        if not self.number_of_times_arm_was_choosen[0] == 0:
            plot_data['1'].append(self.arm_totals[0]/self.number_of_times_arm_was_choosen[0])
        else:
            plot_data['1'].append(0)
            
        if not self.number_of_times_arm_was_choosen[1] == 0:
            plot_data['2'].append(self.arm_totals[1]/self.number_of_times_arm_was_choosen[1])
        else:
            plot_data['2'].append(0)
                
        if not self.number_of_times_arm_was_choosen[2] == 0:
            plot_data['3'].append(self.arm_totals[2]/self.number_of_times_arm_was_choosen[2])
        else:
            plot_data['3'].append(0)
            
        return plot_data
        
    def train(self):
        average_reward = 0
        plot_data = {'total': [], '1': [], '2': [], '3': []}
        for i in range(1, self.n_trials+1):
            arm = self.pick_an_arm()
            self.update_policy()
            self.number_of_times_arm_was_choosen[arm] += 1
            reward = self.get_reward(arm)
            self.update_value(arm, reward)
            
            self.arm_totals[arm] += reward
            average_reward += reward 
            
            plot_data = self.update_average_reward_values(plot_data)
            plot_data['total'].append(average_reward/i)
        
        print('Policy:', self.policy)
        self.plot(plot_data)


def main():
    temps = [0.5, 0.7, 0.9]
    number_of_trials = 1000

    for t in temps:
        sfl = SoftmaxLearning(number_of_trials, 3, t)
        print('temperature = '+str(t))
        sfl.train()
        print('------------------- (:) -------------------')

if __name__ == '__main__':
    main()