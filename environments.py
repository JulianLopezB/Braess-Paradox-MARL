import gym
from gym import spaces

class BraessParadoxEnv(gym.Env):
    
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_agents=4000):
        
        super(BraessParadoxEnv, self).__init__()
        
        self.n_agents = n_agents
        self.cost_params  = {'c1': -n_agents*4.5/4000, 'c2': -n_agents/400}
        
    def reset(self):
        
        self.state = {'step': 0}
        
        for i in range(self.n_agents):
            
            self.state[i] = 'S'
            
        self.done = False
        
        return self.state
    
    def step(self, actions):
        
        info = {}
        for i in range(self.n_agents):
            
            self.state[i] = actions[i]
            
        if self.state['step'] == 0:
            
            self.state['step'] = 1
            T =  sum([a=='A' for a in actions])
            rewards = [ T/self.cost_params['c2'] if a=='A' else self.cost_params['c1'] for a in actions]
            
        else:
            
            T = sum([a=='B' for a in actions])
            rewards = [self.cost_params['c1'] if a=='A' else T/self.cost_params['c2'] for a in actions]
            self.done = True
            
        return self.state, rewards, self.done, info

    
    def render(self, mode='human', close=False):
        
        # Render the environment to the screen
        pass

