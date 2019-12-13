import numpy as np
import matplotlib.pyplot as plt
import math
import random
import operator

class UCB2Learning:
    def __init__(self, n_actions, alpha, n_trials):
        self.number_of_times_action_was_choosen = np.zeros(n_actions)
        self.action_totals = np.zeros(n_actions)
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.t = 0
        self.number_of_actions_so_far = 1
        self.alpha = alpha
        self.n_trials = n_trials
        self.R = np.zeros(n_actions)

    def get_cost_of_action(self, action):
        if action == 0:
            return np.random.normal(2, math.sqrt(0.0625), 1)[0]
        if action == 1:
            return np.random.normal(3.5, math.sqrt(0.25), 1)[0]
        if action == 2:
            return np.random.normal(3.5, math.sqrt(4.5), 1)[0]
    
    def get_delay_of_action(self, action):
        if action == 0:
            return np.random.normal(0, math.sqrt(0.25), 1)[0]
        if action == 1:
            return np.random.uniform(-3, 0.5, 1)[0]
        if action == 2:
            return np.random.normal(-2.5, math.sqrt(0.25), 1)[0]

    def plot(self, plot_data):
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['total'], label='total')
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['1'], label='metro')
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['2'], label='bus')
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['3'], label='taxi')
        plt.legend()
        plt.title('Average Rewards (day by day)')
        plt.xlabel('trial number')
        plt.show()
        
    def update_average_reward_values(self, plot_data):
        if not self.number_of_times_action_was_choosen[0] == 0:
            plot_data['1'].append(self.action_totals[0]/self.number_of_times_action_was_choosen[0])
        else:
            plot_data['1'].append(0)
            
        if not self.number_of_times_action_was_choosen[1] == 0:
            plot_data['2'].append(self.action_totals[1]/self.number_of_times_action_was_choosen[1])
        else:
            plot_data['2'].append(0)
                
        if not self.number_of_times_action_was_choosen[2] == 0:
            plot_data['3'].append(self.action_totals[2]/self.number_of_times_action_was_choosen[2])
        else:
            plot_data['3'].append(0)
            
        return plot_data

    def play_action_and_get_reward(self, action):
        cost = self.get_cost_of_action(action)*1000
        delay = self.get_delay_of_action(action)
        if delay < 0:
            money_because_of_time_diff = 1000*abs(delay)
        else:
            money_because_of_time_diff = -1500*abs(delay)
        total_reward = money_because_of_time_diff - cost 
        return total_reward

    def update_value(self, action, reward):
        lr = 1/self.number_of_times_action_was_choosen[action]
        self.Q[action] = self.Q[action] + lr * (reward - self.Q[action])


    def play_each_action_once(self):
        for action in range(self.n_actions):
            r = self.play_action_and_get_reward(action)
            self.number_of_times_action_was_choosen[action] += 1
            self.action_totals[action] += r
            self.Q[action] = r

    def calculate_tha_r(self, r):
        return math.ceil(math.pow(1+self.alpha, r))

    def pick_action(self):
        action_value = {}
        for action in range(self.n_actions):
            r_i = self.R[action]
            t_r = self.calculate_tha_r(r_i)
            dnum_value = 2*t_r
            ln_value = math.log((self.number_of_actions_so_far*math.e)/t_r)
            a_n_r = math.sqrt((1+self.alpha)*(ln_value)/dnum_value)
            action_value[action] = self.Q[action] + a_n_r
        sorted_action_value = sorted(action_value.items(), key=operator.itemgetter(1))
        return sorted_action_value[-1][0]

    def play_action_j(self, action):
        number_of_times_to_play = int(self.calculate_tha_r(self.R[action]+1) - self.calculate_tha_r(self.R[action]))+1
        for _ in range(number_of_times_to_play):
            self.number_of_actions_so_far += 1
            r = self.play_action_and_get_reward(action)

            self.number_of_times_action_was_choosen[action] += 1
            self.action_totals[action] += r

            self.update_value(action, r)          

    def train(self):
        average_reward = 0
        plot_data = {'total': [], '1': [], '2': [], '3': []}
        self.play_each_action_once()

        for i in range(1, self.n_trials+1):
            action = self.pick_action()
            self.play_action_j(action)
            self.R[action] += 1

            plot_data = self.update_average_reward_values(plot_data)
            plot_data['total'].append(average_reward/i)
        
        print("Q Values:", self.Q)
        print("Number of plays:", self.number_of_times_action_was_choosen)
        self.plot(plot_data)

    

# actions and meanings in this Q
# 0 metro
# 1 bus
# 2 taxi

def main():
    for alpha_value in [0, 0.001, 0.01]:
        print("Alpha:", alpha_value)
        UCB2Learning(3, alpha=alpha_value, n_trials = 1000).train()

if __name__ == '__main__':
    main()