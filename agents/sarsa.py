"""
SARSA Agent.
"""

import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from agents.agent import Agent

import numpy as np

class Sarsa(Agent):
    """ Sarsa algorithm """
    
    def __init__(self, env):
        self.Q = np.zeros((env.width,env.height,len(env.ACTIONS)))
        self.alpha = 0.5
        self.gamma = 1.0
        self.epsilon = 0.1
        self.env = env
    
    def policy_action(self, state, eps=None):
        """ returns sampled policy action at given state."""
        if np.random.rand() < (eps if eps is not None else self.epsilon):
            return np.random.randint(len(self.env.ACTIONS))
        else:
            x,y = state
            return np.random.choice(np.flatnonzero(self.Q[x,y] == self.Q[x,y].max()))        
        
    def show_policy(self):
        # display our policy        

        plt.imshow(self.env.tile)
        
        self.env.reset()
        x,y = self.env.state
        
        history = [(x,y,0)]
        
        for i in range(100):
            # take determinstic policy
            a = self.policy_action(self.env.state,0) 
            self.env.step(int(a))
            x,y = self.env.state
            history.append((x, y, i+1))
            if (x,y) == self.env.goal:
                print("Policy reached goal in {} steps".format(i+1))
                break
        else:
            print("Failed to find path to goal after {} steps.".format(i))

        X = [x for (x,y,c) in history]
        Y = [y for (x,y,c) in history]
        C = [c for (x,y,c) in history]
                        
        plt.scatter(Y,X,c='red',s = 25.0/self.env.width)
        plt.show()

    
    def evaluate(self, max_steps=100, deterministic=False):
        """ Evaluates the agents performance with a potentialy determanistic
            policy.
            Returns number of steps taken and reward.
        """
        self.env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            a = self.policy_action(self.env.state, eps=0 if deterministic else self.epsilon)
            (new_state, reward, done, info) = self.env.step(int(a))
            total_reward += reward
            if done:
                break
        return step+1, total_reward
            
            
    def train(self, iterations=100, max_steps=100):
        """ Train the agent.
            Returns evaluations at various intervals.
        """
        
        EVAL_EVERY = 10
        
        history = []
        for n in range(iterations):
            obs = self.env.reset()
            a = self.policy_action(obs)
            for step in range(max_steps):
                x,y = self.env.state                
                (new_obs, reward, done, info) = self.env.step(int(a))
                x_, y_ = new_obs
                a_ = self.policy_action(new_obs)
                                                
                self.Q[x,y,a] += self.alpha * (reward + self.gamma * self.Q[x_, y_, a_] - self.Q[x,y,a])
                x,y,a = x_, y_, a_
                                
                if done:
                    break
                
            # do a determanistic evaluation.
            if n % EVAL_EVERY == 0:
                steps_taken, total_reward = self.evaluate(max_steps)
                history.append((n, steps_taken, total_reward))               
            
        return history