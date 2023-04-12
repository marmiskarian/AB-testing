# Epsilon Greedy and Thompson Sampling Implementations

This repository contains Python implementations of two popular multi-armed bandit algorithms - **Epsilon Greedy** and **Thompson Sampling**. These algorithms are commonly used in online advertising, recommendation systems, and other areas where one needs to optimize decision-making under uncertainty.

##### *Epsilon Greedy*
*Epsilon Greedy* is a simple algorithm that balances exploration and exploitation by selecting the best action with probability 1-epsilon and a random action with probability epsilon.

##### *Thompson Sampling*
*Thompson Sampling*, on the other hand, is a probabilistic algorithm that chooses actions based on their estimated probability of being optimal. The algorithm maintains a distribution over the expected reward for each action, and samples an action from this distribution at each step.

Both algorithms are implemented using Python, NumPy, and Matplotlib. The code is well-commented and should be easy to understand and modify.

### How to use
To use the implemented algorithms for multi-armed bandit problem, create an instance of `EpsilonGreedy` or `ThompsonSampling` class with the desired parameters and call the corresponding `experiment()` method to run the experiment. The output is a list of rewards obtained at each step, which can be used to compare the performance of the two algorithms using the provided `comparison()` function.

Both classes provide plotting and report methods to visualize the learning process and to generate a summary report of the experiment. Additionally, the output of the experiments are saved in csv files for future reference.

This repository is intended to serve as a starting point for anyone interested in implementing and experimenting with multi-armed bandit algorithms.

Note: Logging is not implemented in the code yet.