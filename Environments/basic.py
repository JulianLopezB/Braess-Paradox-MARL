import gym
from gym.spaces import Dict, Discrete, Box

class BraessParadoxEnv():
    
    metadata = {'render.modes': ['human']}

    def __init__(self, n_agents=4000, social_welfare_type='utilitarian'):
        
        super(BraessParadoxEnv, self).__init__()
        self.n_agents = n_agents
        self.cost_params  = {'c1': -n_agents*45/4000, 'c2': -n_agents/40}
        self.social_welfare_type = social_welfare_type

    def reset(self):
        
        self.state = {'step': 0, 'actions': {}}
        for i in range(self.n_agents):
            self.state['actions'][i] = 'S'
        self.done = False
        
        return self.state
    
    def step(self, actions):

        info = {}
        for i in range(self.n_agents):
            self.state['actions'][i] = actions[i]
        if self.state['step'] == 0:
            self.state['step'] = 1
            T =  sum([a=='A' for a in actions])
            rewards = [ T/self.cost_params['c2'] if a=='A' else self.cost_params['c1'] for a in actions]
        else:
            T = sum([a=='B' for a in actions])
            rewards = [self.cost_params['c1'] if a=='A' else T/self.cost_params['c2'] for a in actions]
            self.done = True
        rewards = self.social_welfare(rewards)

        return self.state, rewards, self.done, info

    
    def social_welfare(self, rewards):

        # Utilitarian Social Welfare
        if self.social_welfare_type == 'utilitarian':
            rewards = [sum(rewards)/(self.n_agents)]*(self.n_agents)
        # Rawls Social Welfare
        elif self.social_welfare_type == 'rawlsian':
            rewards = [min(rewards)]*(self.n_agents)

        return rewards

    def render(self, mode='human', close=False):
        
        # Render the environment to the screen
        pass


class BraessParadoxGymEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self, n_agents=4000, social_welfare_type='utilitarian'):
        
        super(BraessParadoxGymEnv, self).__init__()
        self.n_agents = n_agents
        self.cost_params  = {'c1': -n_agents*45/4000, 'c2': -n_agents/40}
        self.social_welfare_type = social_welfare_type
        self.action_space = Discrete(2)
        self.state_space = Dict({"position": Discrete(2),
                          "velocity": Box(low=0, high=3, shape=(self.n_agents,), dtype=int) 
                         })
        self.reset()
        
    def reset(self):
        
        self.state = {'step': 0, 'positions': {}}
        for i in range(self.n_agents):
            self.state['positions'][i] = 0
        self.done = False
        
        return self.state
    
    def step(self, actions):

        info = {}
        for i in range(self.n_agents):

            # Transitions
            if self.state['positions'][i] == 0:
                if actions[i] == 0:
                    self.state['positions'][i] = 1
                if actions[i] == 1:
                    self.state['positions'][i] = 2
            else:
                self.state['positions'][i] = 3

        if self.state['step'] == 0:
            self.state['step'] = 1
            T =  sum([a==0 for a in actions])
            rewards = [ T/self.cost_params['c2'] if a==0 else self.cost_params['c1'] for a in actions]
        else:
            T = sum([a==1 for a in actions])
            rewards = [self.cost_params['c1'] if a==0 else T/self.cost_params['c2'] for a in actions]
            self.done = True
        rewards = self.social_welfare(rewards)

        return self.state, rewards, self.done, info

    
    def social_welfare(self, rewards):

        # Utilitarian Social Welfare
        if self.social_welfare_type == 'utilitarian':
            rewards = [sum(rewards)/(self.n_agents)]*(self.n_agents)
        # Rawls Social Welfare
        elif self.social_welfare_type == 'rawlsian':
            rewards = [min(rewards)]*(self.n_agents)

        return rewards

    def render(self, mode='human', close=False):
        
        # Render the environment to the screen
        pass