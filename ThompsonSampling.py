from Bandit import Bandit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class ThompsonSampling(Bandit):
    """
    An implementation of the Thompson Sampling algorithm for multi-armed bandit problems.

    Inherits from the Bandit class.
    """
    
    def __init__(self, true_mean):
        """
        Constructor for the ThompsonSampling class.
        """
        self.true_mean = true_mean
        self.m = 0
        self.lambda_ = 1
        self.tau = 1
        self.N = 0
        self.sum_x = 0
        
    def __repr__(self):
        """
        String representation of the class.
        """
        return f"A Bandit with {self.true_mean} Win Rate"

    def pull(self):
        """
        Samples a reward from the bandit using its true mean.
        """
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def sample(self):
        """
        Samples a reward from the bandit using its posterior mean.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        """
        Updates the bandit's posterior mean and precision using the reward received.
        """
        self.m = (self.tau * x + self.lambda_ * self.m) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
    
    def experiment(self, bandit_rewards, N):
        """
        Runs the Thompson Sampling algorithm on a set of bandits.
        """
        bandits = [ThompsonSampling(m) for m in bandit_rewards]
        
        sample_points = [100, 1000, 2000, 5000, 10000, 19999]
        
        # empty array to later add the rewards for inference plots
        t_rewards = np.empty(N)
        t_selected_bandit = []
        
        for i in range(N):
            j = np.argmax([b.sample() for b in bandits]) #taking the highest mean position
            
            # make some plots
            if i in sample_points:
                self.plot_bandit_distributions(bandits, i)
            
            # pull the chosen bandit and get the output
            x = bandits[j].pull()

            # increases N by 1, updates lambda and calculates the estimate of the m
            bandits[j].update(x)
            
            # add the reward to the data
            t_rewards[i] = x
            
            # Add the selected bandit to the list
            t_selected_bandit.append(j)
        
        all_bandits = pd.DataFrame({"Bandit" : t_selected_bandit, "Reward" : t_rewards, "Algorithm" : "Thompson Sampling"})
        all_bandits.to_csv("./csv/ThompsonSampling_All.csv", index=False)

        return bandits, t_rewards
    
    def plot_learning_process(self, bandit_rewards, t_rewards, N):
        """
        Plots the win rate and optimal win rate against the number of trials.
        """
        cumulative_rewards = np.cumsum(t_rewards)
        win_rates = cumulative_rewards / (np.arange(N) + 1)
        
        plt.figure(figsize=(10, 8))
        plt.plot(win_rates, label="Win Rate")
        plt.plot(np.ones(N)*np.max(bandit_rewards), label='Optimal Win Rate')
        plt.legend()
        plt.title("Win Rate Convergence Thompson Sampling")
        plt.xlabel("Number of Trials")
        plt.ylabel("Average Reward")
        plt.show()

    def plot_bandit_distributions(self, bandits, trial):
        """
        Plots the distribution of each bandit after a given number of trials.
        """
        x = np.linspace(-3, 6, 200)
        for b in bandits:
            y = norm.pdf(x, b.m, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label = f"real mean: {b.true_mean:.4f}, num plays: {b.N}")
            plt.title(f"Bandit distributions after {trial} trials")
        plt.legend()
        plt.show()
        
    def report(self, bandits, t_rewards, N):
        """
        Creates a report with statistics such as total reward earned, average reward,
        and the overall win rate, and saves the output in a csv file.
        """
        df = pd.DataFrame()
        for b in bandits:
            df["Bandit"] = [b for b in bandits]
            df["Reward"] = [b.m for b in bandits]
            df["Algorithm"] = "ThompsonSampling"
        
        print(f"Total Reward Earned: {t_rewards.sum()}")
        print(f"Average Reward: {np.mean(t_rewards)}")
        print(f"Overall Win Rate: {t_rewards.sum() / N}")
        print(f"Number of times selected each bandit: {[b.N for b in bandits]}")
        
        df.to_csv("./csv/ThompsonSampling_Last.csv", index=False)
        
        return df