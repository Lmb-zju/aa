# Import required libraries
import numpy as np
import matplotlib.pyplot as plt


# Define Action class
class Actions:
    def __init__(self, m):
        self.m = m
        self.mean = 0
        self.N = 0

    # Choose a random action (returns reward)
    def choose(self):
        return np.random.randn() + self.m

    # Update the action-value estimate
    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x


def run_experiment(m1, m2, m3, eps, N):
    actions = [Actions(m1), Actions(m2), Actions(m3)]
    data = np.empty(N)

    for i in range(N):
        # epsilon greedy
        p = np.random.random()
        if p < eps:
            j = np.random.choice(3)
        else:
            j = np.argmax([a.mean for a in actions])

        # Get reward and update action
        x = actions[j].choose()
        actions[j].update(x)
        data[i] = x  # Store reward for plotting

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    # Plot results
    plt.plot(cumulative_average)
    plt.plot(np.ones(N) * m1, label='m1')
    plt.plot(np.ones(N) * m2, label='m2')
    plt.plot(np.ones(N) * m3, label='m3')
    plt.xscale('log')
    plt.legend()
    plt.show()

    # Print final means
    for a in actions:
        print(a.mean)

    return cumulative_average


if __name__ == '__main__':
    c_1 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000)
    c_05 = run_experiment(1.0, 2.0, 3.0, 0.05, 100000)
    c_01 = run_experiment(1.0, 2.0, 3.0, 0.01, 100000)
