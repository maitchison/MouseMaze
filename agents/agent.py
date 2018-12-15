"""
Collection of reinforcement learning agents
"""


class Agent():
    
    def __init__(self, name="Agent"):
        self.name = name
    
    def act(self, observation, reward):
        raise NotImplementedError()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class RandomAgent(Agent):
    
    def __init__(self, action_space):
        super().__init__("Random Agent")
        self.action_space = action_space

    def act(self, observation, reward):
        return self.action_space.sample()
