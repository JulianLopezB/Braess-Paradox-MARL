# Braess Paradox Environment
An implementation of the Breaess Paradox for multi-agent reinforcement learning experiments.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Braess_paradox_road_example.svg/500px-Braess_paradox_road_example.svg.png)

## About
Discovered by German mathematician Dietrich Braess in 1968, the paradox (aka The Network Paradox)  is the observations that, counterintuitively, adding a new road to a road network not only may not help but it can actually slow down overall traffic.
This implementation allows to experiment with multi-agent reinforcemnet learning on an environment that emulates the Braess Paradox. 

## The Envioronment

### State Space

### Action Space

### Social Welfares

- Utilitarian

- Rawlsian


## Example usage

Basic usage:
```python
from environments import BraessParadoxEnv
from agents import QAgent

env = BraessParadoxEnv(n_agents=400)

params = {'lr': 0.005, 
          'gamma': 0.999, 
          'eps_start': 1.0, 
          'eps_end': 0.001,
          'eps_dec': 0.995}

agents = [QAgent(**params, id_agent=i) for i in range(n_agents)]

new_state = env.reset()


actions = [agents[i].choose_action(new_state) for i in range(env.n_agents)]

new_state, rewards, done, info = env.step(actions)
```

or, if you want to run a whole episode:
```python
done = False
state = env.reset()

while not done:

    actions = [agents[i].choose_action(state) for i in range(n_agents)]
    new_state, rewards, done, info = env.step(actions)
```

for more details see this [example](https://github.com/JulianLopezB/Braess-Paradox-MARL/blob/main/notebooks/Braess-Paradox.ipynb)

## Roadmap:

- ### Algorithms (Players):
- [X] Q Learning Agent 
- [] Policy Gradient
- [] Dyna
- [] MonteCarlo
- [] Sarsa
- [] DQN Learning Agent
- [] Minimax-Q Learning Agent
- [] Correlated-Q Learninh Agent
- [] Nash Q-learning Agent

- ### Environment and Metrics
- [] Add Price of Anarchy

- ### Tooling
- [] Generalize Traffic Networks
- [] Tools to render states