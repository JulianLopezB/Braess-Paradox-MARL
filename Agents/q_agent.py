import gym
import numpy as np
from random import random

class QAgent():
    ''' 
    Q-Learning algorithm.
    Agent only sees his position

    '''
    def __init__(self, id_agent, lr, gamma, eps_start, eps_end, eps_dec):
        
        self.id_agent = id_agent
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.Q = {}

        self.init_Q()

    def init_Q(self):

        for step in {0, 1}:
            for position in {0, 1, 2, 3}:
                for action in {0, 1}:
                    self.Q[(position, step, action)] = 0.0

    def choose_action(self, state):

        position = state['positions'][self.id_agent]
        step = state['step']
        Q_ = np.array([self.Q[(position, step, a)] for a in [0, 1]])
        explore = random() < self.epsilon
        if step == 0:
            if explore:
                action = np.random.choice([0, 1])
            else:
                action = int(np.argmax(Q_))
        else:
            if state['positions'][self.id_agent] == 1:
                if explore:
                    action = np.random.choice([0, 1])
                else:
                    action = int(np.argmax(Q_))
            elif state['positions'][self.id_agent] == 2:
                action = 1
            else:
                pass
        return  action

    def decrement_epsilon(self):

        self.epsilon = self.epsilon*self.eps_dec if self.epsilon>self.eps_min\
                       else self.eps_min

    def learn(self, state, action, reward, state_):
        position = state['positions'][self.id_agent]
        step = state['step']
        q_actions = np.array([self.Q[(position, step, a)] for a in [0, 1]])
        a_q_max = int(np.argmax(q_actions))
        self.Q[(position, step, action)] += self.lr*(reward + self.gamma*self.Q[(position, step, a_q_max)] - self.Q[(position, step, action)])
        self.decrement_epsilon()

