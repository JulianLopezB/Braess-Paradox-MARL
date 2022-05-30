import numpy as np
from matplotlib import pyplot as plt

def plot_scores(scores, sample_agents=10):
    window = int(len(scores[0])*0.1)
    print(window)
    for x in range(sample_agents):
        plt.plot([np.mean(scores[x][i-window:i]) for i in range(window, len((scores[x])))])
    plt.title(f'Rewards of random sample size {sample_agents} agents')
    plt.show()



def plot_actions(actions_list):
    n_agents = len(actions_list[0][0])
    plt.plot([[sum([a == 0 for a in step])/n_agents for step in episode][0] for episode in actions_list], label='Step 1')
    plt.plot([[sum([a == 0 for a in step])/n_agents for step in episode][1] for episode in actions_list], label='Step 2')
    plt.title("Fraction of agents playing action 'A'")
    plt.legend()

def render(actions_list):
    pass