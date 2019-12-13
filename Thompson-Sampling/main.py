import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import beta

class ThompsonLearning:
    def __init__(self, n_actions):
        self.n_trials = None
        self.n_actions = n_actions
        self.alpha = np.ones(n_actions)
        self.beta = np.ones(n_actions)
        self.arm_totals = np.zeros(n_actions)
        self.number_of_times_arm_was_choosen = np.zeros(n_actions)
        
    def generate_reward(self, arm):
        if arm==0:
            return np.random.normal(3, math.sqrt(25), 1)[0]
        if arm == 1:
            random_val = random.uniform(0, 1)
            if random_val < 0.7:
                return np.random.normal(15, math.sqrt(16), 1)[0]
            else:
                return np.random.uniform(-15,10,1)[0]
        if arm == 2:
            return np.random.uniform(-15,20,1)[0]
        if arm == 3:
            random_val = random.uniform(0, 1)
            if random_val < 0.6:
                return np.random.uniform(-5,25,1)[0]
            else:
                return np.random.normal(-5, math.sqrt(25), 1)[0] 
        
        
    def get_reward(self, arm):
        r = self.generate_reward(arm) 
        if r>0:
            return 1
        return 0
    
    def sample_beta_dist(self, arm):
        return np.random.beta(self.alpha[arm], self.beta[arm], 1)[0]
    
    def update_estimations(self, arm, reward):
        self.alpha[arm] = self.alpha[arm] + reward
        self.beta[arm] = self.beta[arm] + 1 - reward
        
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
            
        if not self.number_of_times_arm_was_choosen[3] == 0:
            plot_data['4'].append(self.arm_totals[3]/self.number_of_times_arm_was_choosen[3])
        else:
            plot_data['4'].append(0)
            
        return plot_data
    
    def plot(self, plot_data):
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['total'], label='total')
        plt.legend(loc='best')
        plt.title('total average reward')
        plt.xlabel('trial number')
        plt.show()
        
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['1'], label='arm 1')
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['2'], label='arm 2')
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['3'], label='arm 3')
        plt.plot([x for x in range(1, self.n_trials+1)], plot_data['4'], label='arm 4')
        plt.legend(loc='best')
        plt.title('average reward of different arms')
        plt.xlabel('trial number')
        plt.show()
        
    def draw_beta_distributions(self):
        x_values = [x/100 for x in range(100)]
        for i in range(self.n_actions):
            dist = beta(self.alpha[i], self.beta[i])
            plt.plot(x_values, dist.pdf(x_values), label='arm '+str(i+1))
        plt.legend(loc='best')
        plt.title('Beta Distribution')
        plt.xlabel('x')
        plt.show()
    
        
    def print_belief_values(self):
        for i in range(self.n_actions):
            print('(alpha({}), beta({})) = ({}, {})'.format(i+1, i+1, self.alpha[i], self.beta[i]))
    
    def train(self, num_iterations):
        average_reward = 0
        self.n_trials = num_iterations
        plot_data = {'total': [], '1': [], '2': [], '3': [], '4':[]}
        for _ in range(1, num_iterations+1):
            all_mues = []
            for i in range(self.n_actions):
                all_mues.append(self.sample_beta_dist(i))
            
            all_mues = np.array(all_mues)
            a_t = np.argmax(all_mues) 
            r_t = self.get_reward(a_t)
            
            self.update_estimations(a_t, r_t)
            
            self.number_of_times_arm_was_choosen[a_t] += 1
            self.arm_totals[a_t] += r_t
            average_reward += r_t 
            
            plot_data = self.update_average_reward_values(plot_data)
            plot_data['total'].append(average_reward/i)
           
        self.plot(plot_data)
        self.print_belief_values()
        self.draw_beta_distributions()

def main():
    tl = ThompsonLearning(4)
    for trial_number in [100, 1000, 10000, 100000]:
        print("Number of trials:", trial_number)
        tl.train(trial_number)
        print('------------------- (:) -------------------')

if __name__ == '__main__':
    main()