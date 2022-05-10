# Braess Paradox Environment
An implementation of the Breaess Paradox for multi-agent reinforcement learning experiments.

[bp-wikipedia](https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Braess_paradox_road_example.svg/500px-Braess_paradox_road_example.svg.png)

## About
Discovered by German mathematician Dietrich Braess in 1968, the paradox (aka The Network Paradox)  is the observations that, counterintuitively, adding a new road to a road network not only may not help but it can actually slow down overall traffic.
This implementation allows to experiment with multi-agent reinforcemnet learning on an environment that emulates the Braess Paradox. 

## Example usage
Basic usage:
```python
from environments import BraessParadoxEnv
from agents import QAgent

env = BraessParadoxEnv(n_agents=400)
agents = [QAgent(**params, id_agent=i) for i in tqdm(range(n_agents))]

new_state = env.reset()


actions = [agents[i].choose_action(new_state) for i in tqdm(range(env.n_agents))]

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