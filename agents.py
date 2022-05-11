import gym
import numpy as np
from random import random
class QAgent():
    
    def __init__(self, id_agent, lr, gamma, n_agents, eps_start, eps_end, eps_dec):
        
        self.id_agent = id_agent
        self.lr = lr
        self.gamma = gamma
        self.n_agents = n_agents
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.Q = {}

        self.init_Q()

    def init_Q(self):
        
        for step in {0, 1}:
            for position in {'S', 'A', 'B'}:
                for action in {'A', 'B'}:
                    self.Q[(position, step, action)] = 0.0

    def choose_action(self, state):
        
        position = state[self.id_agent]
        step = state['step']
        

        Q_ = np.array([self.Q[(position, step, a)] for a in ['A', 'B']])
        explore = random() < self.epsilon
        
        if step == 0:
            
            if explore:
                action = np.random.choice(['A', 'B'])
            else:
                action = ['A', 'B'][int(np.argmax(Q_))]
            
        else:
            if state[self.id_agent] == 'A':
                if explore:
                    action = np.random.choice(['A', 'B'])
                    
                else:
                    action = ['A', 'B'][int(np.argmax(Q_))]
            else:
                action = state[self.id_agent]
            
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon*self.eps_dec if self.epsilon>self.eps_min\
                       else self.eps_min

    def learn(self, state, action, reward, state_):
        
        position = state[self.id_agent]
        step = state['step']

        actions = np.array([self.Q[(position, step, a)] for a in ['A', 'B']])
        
        a_max = ['A', 'B'][int(np.argmax(actions))]

        self.Q[(position, step, action)] += self.lr*(reward +
                                        self.gamma*self.Q[(position, step, a_max)] -
                                        self.Q[(position, step, action)])
        self.decrement_epsilon()