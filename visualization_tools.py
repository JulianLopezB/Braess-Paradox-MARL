import numpy as np
from matplotlib import pyplot as plt

def plot_scores(scores):
    for x in range(10):
        plt.plot([np.mean(scores[x][i-100:i]) for i in range(100, len((scores[x])))])
    plt.show()

def plot_actions(actions_list):
    n_agents = len(actions_list[0][0])
    plt.plot([[sum([a == 'A' for a in step])/n_agents for step in episode][0] for episode in actions_list], label='Step 1')
    plt.plot([[sum([a == 'A' for a in step])/n_agents for step in episode][1] for episode in actions_list], label='Step 2')
    plt.title("Fraction of agents playing action 'A'")
    plt.legend()

def render(actions_list):
    pass