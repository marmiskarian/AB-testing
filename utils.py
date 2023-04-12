import matplotlib.pyplot as plt

def comparison(epsilon_rewards, thompson_rewards):
    """
    Plot and compare the performance of two algorithms, Epsilon Greedy and Thompson Sampling,
    based on their cumulative rewards and mean reward per trial.
    """
    # Plot cumulative rewards for each algorithm
    plt.figure(figsize=(10, 8))
    plt.plot(epsilon_rewards, label='Epsilon Greedy', alpha=0.75)
    plt.plot(thompson_rewards, label='Thompson Sampling', alpha=0.75)
    plt.title('Comparison of Epsilon Greedy and Thompson Sampling', fontsize=20)
    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('Cumulative Reward', fontsize=14)
    plt.legend()
    plt.show()

    # Plot mean reward per round for each algorithm
    epsilon_mean = [sum(epsilon_rewards[:i+1])/(i+1) for i in range(len(epsilon_rewards))]
    thompson_mean = [sum(thompson_rewards[:i+1])/(i+1) for i in range(len(thompson_rewards))]
    plt.figure(figsize=(10, 8))
    plt.plot(epsilon_mean, label='Epsilon Greedy')
    plt.plot(thompson_mean, label='Thompson Sampling')
    plt.title('Comparison of Epsilon Greedy and Thompson Sampling', fontsize=20)
    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('Mean Reward', fontsize=14)
    plt.legend()
    plt.show()