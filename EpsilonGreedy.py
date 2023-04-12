from Bandit import Bandit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EpsilonGreedy(Bandit):
    """
    An implementation of the Epsilon Greedy algorithm for multi-armed bandit problems.

    Inherits from the Bandit class.
    """
    
    def __init__(self, reward):
        """
        Constructor for the EpsilonGreedy class.
        """
        self.reward = reward
        self.reward_estimate = 0
        self.N = 0
    
    def __repr__(self):
        """
        String representation of the class.
        """
        return f'A Bandit with {self.reward} Reward'
    
    def pull(self):
        """
        Pulls the arm of the bandit and generate a random reward.
        """
        return np.random.randn() + self.reward
    
    def update(self, x):
        """
        Updates the reward estimate based on the current reward obtained.
        """
        self.N += 1
        self.reward_estimate = (1 - 1.0/self.N) * self.reward_estimate + 1.0/ self.N * x

    def experiment(self, bandit_rewards, t, N):
        """
        Runs the Epsilon Greedy algorithm on a set of bandits.
        """
        bandits = [EpsilonGreedy(reward) for reward in bandit_rewards]
        
        num_times_explored = 0
        num_times_exploited = 0
        num_optimal = 0
        optimal_j = np.argmax([b.reward for b in bandits])
        print(f'optimal bandit index: {optimal_j}')
        
        # empty array to later add the rewards for inference plots
        eg_rewards = np.empty(N)
        eg_selected_bandit = []
        eps = 1/t

        for i in range(N):
            #generating a random number 
            p = np.random.random() 
            
            # if the random number is smaller than eps we explore a random bandit
            if p < eps:  
                num_times_explored += 1
                j = np.random.choice(len(bandits)) 
            else:
                # if the random number is bigger than eps we explore the bandit with the highest current reward
                num_times_exploited += 1
                j = np.argmax([b.reward_estimate for b in bandits])
            
            # pull the chosen bandit and get the output
            x = bandits[j].pull()
            
            # increases N by 1 and calculates the estimate of the reward
            bandits[j].update(x) 
            
            # if j is the actual optimal bandit, the optimal bandit count increments by 1
            if j == optimal_j:
                num_optimal += 1
            
            # add the selected bandit to the list of selected bandits
            eg_selected_bandit.append(j)
            
            # add the reward to the data
            eg_rewards[i] = x
            
            # increase t, i.e., decrease the probability of choosing suboptimal (random) bandit
            t += 1
            eps = 1/t
        
        estimated_avg_rewards=[round(b.reward_estimate, 3) for b in bandits]
        print(f'Estimated average reward where epsilon= {eps}:---{estimated_avg_rewards}')
        
        all_bandits = pd.DataFrame({"Bandit" : eg_selected_bandit, "Reward" : eg_rewards, "Algorithm" : "Epsilon Greedy"})
        all_bandits.to_csv("./csv/EpsilonGreedy_All.csv", index=False)
        
        return bandits, eg_rewards, num_times_explored, num_times_exploited, num_optimal
    
    def plot_learning_process(self, bandit_rewards, eg_rewards, N):
        """
        Plots the win rate and optimal win rate against the number of trials.
        """        
        cumulative_rewards = np.cumsum(eg_rewards)
        win_rates = cumulative_rewards / (np.arange(N) + 1)
        
        plt.figure(figsize=(10, 8))
        plt.plot(win_rates, label="Win Rate")
        plt.plot(np.ones(N)*np.max(bandit_rewards), label='Optimal Win Rate')
        plt.legend()
        plt.title("Win Rate Convergence Epsilon-Greedy")
        plt.xlabel("Number of Trials")
        plt.ylabel("Average Reward")
        plt.show()

    def report(self, bandits, eg_rewards, num_times_explored, num_times_exploited, num_optimal, N):
        """
        Creates a report with statistics such as mean estimates, total reward earned,
        and the number of times a bandit was explored or exploited, and saves the output in a csv file.
        """
        df = pd.DataFrame()
        for i, b in enumerate(bandits):
            print(f"Bandit {i} Mean Estimate: {b.reward_estimate :.4f}")
            df["Bandit"] = [b for b in bandits]
            df["Reward"] = [b.reward_estimate for b in bandits]
            df["Algorithm"] = "EpsilonGreedy"
    
        print(f"\nTotal Reward Earned: {eg_rewards.sum()}")
        print(f"Average Reward: {np.mean(eg_rewards)}")
        print(f"Overall Win Rate: {eg_rewards.sum() / N :.4f}\n")
        print(f"# of explored: {num_times_explored}")
        print(f"# of exploited: {num_times_exploited}")
        print(f"# of times selected the optimal bandit: {num_optimal}")
        
        df.to_csv("./csv/EpsilonGreedy_Last.csv", index=False)
        
        return df