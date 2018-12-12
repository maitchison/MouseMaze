"""
Maze enviroments
"""

import numpy as np
import matplotlib.pyplot as plt
import kruskal
import gym
from utilities import clip, smooth, bits_to_int, int_to_bits
from collections import deque
from gym import error, spaces, utils

class Maze_Base(gym.Env):
    """ Base class for maze environments.
        Handles generation etc.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTIONS = ['north', 'south', 'east', 'west']
    DELTAS = [(0, -1), (0, +1), (-1, 0), (+1, 0)]

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.tile = np.zeros((width, height))
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.state = (-1, -1)

    # ----------------------------------
    # Open AI interface
    # ----------------------------------

    def reset(self):
        """ reset environment and return initial observation. """
        self.state = self.get_random_initial_position()
        return self.get_observation()

    def render(self, mode="human", close=False):
        # not implemented yet...
        output = self.tile
        x, y = self.state
        output[x, y] = -1
        return output

    def step(self, action):
        assert isinstance(action, int), "Action must be of type int."
        self.move(*self.DELTAS[action])
        reward = 1 if self.at_goal() else -0.1
        done = (reward == 1)
        info = {}
        return self.get_observation(), reward, done, info

    # ----------------------------------
    # General
    # ----------------------------------

    def get_random_initial_position(self):
        """ returns a random start location. """
        potential_states = []
        for x in range(self.height):
            for y in range(self.width):
                if self.tile[x,y] == 0:
                    potential_states.append((x,y))
        if len(potential_states) == 0:
            raise Exception("No initial starting position found.")
        return potential_states[np.random.randint(len(potential_states))]

    def get_observation(self):
        raise NotImplementedError

    def at_goal(self):
        return self.state == self.goal

    def move(self, dx, dy):
        x, y = self.state
        x += dx
        y += dy
        if self.can_move_to(x, y):
            self.state = (x, y)
        else:
            # invalid move
            pass

    def can_move_to(self, x, y):
        return 0 < x < self.width and 0 < y < self.width and y < self.height and self.tile[x, y] % 2 != 1

    def calculate_min_path(self, start):
        """ Calculates minimum distance from start to finish. """
        q = deque()
        q.append((*start, 0))
        visited = set(start)
        while q:
            x, y, cost = q.popleft()

            if (x, y) == self.goal:
                return cost

            for dx, dy in self.DELTAS:

                if (x + dx, y + dy) not in visited and self.can_move_to(x + dx, y + dy):
                    q.append((x + dx, y + dy, cost + 1))
                    visited.add((x + dx, y + dy))

        return -1

    def generate_random(self, seed=None):
        """ generate a random maze using Kruskals algorithm"""

        assert self.width >= 3 and \
               self.height >= 3 and \
               self.width % 2 == 1 and \
               self.height % 2 == 1, \
            "Width and height must be >=3 and odd."

        if seed is not None:
            np.random.seed(seed)

        # The idea is to use kruskals with random weights on a smaller grid then expand the grid out
        # so that there are rooms for the walls inbetween the tiles.

        h_width = self.width // 2
        h_height = self.height // 2

        g = kruskal.Graph(h_width * h_height)

        def in_bounds(x, y):
            return x >= 0 and y >= 0 and x < h_width and y < h_height

        def add_edge(g, x1, y1, x2, y2, cost):
            if in_bounds(x1, y1) and in_bounds(x2, y2):
                g.addEdge(x1 + y1 * h_width, x2 + y2 * h_width, cost)
                g.addEdge(x2 + y2 * h_width, x1 + y1 * h_width, cost)

        for x in range(h_width):
            for y in range(h_height):
                for dx, dy in self.DELTAS:
                    add_edge(g, x, y, x + dx, y + dy, np.random.randint(1, 10))

        graph = g.KruskalMST()

        self.tile = np.ones((self.width, self.height))

        goal = (np.random.randint(h_width), np.random.randint(h_height))

        # save goal location for later
        self.goal = (goal[0] * 2 + 1, goal[1] * 2 + 1)

        # fill in rooms
        for x in range(h_width):
            for y in range(h_height):
                mask = 0
                if ((x, y) == goal): mask += 2
                self.tile[x * 2 + 1, y * 2 + 1] = mask

        # convert from graph to walls
        for head, tail, weight in graph:
            x1, y1 = head % h_width, head // h_width
            x2, y2 = tail % h_width, tail // h_width
            dx = x2 - x1
            dy = y2 - y1
            self.tile[x1 * 2 + 1 + dx, y1 * 2 + 1 + dy] = 0

            # by default the maze will have only one path to the goal, we want multiple
        # non-optimal paths so we delete some walls
        for n in range(h_width * h_height):
            x = np.random.randint(1, self.width - 1)
            y = np.random.randint(1, self.height - 1)
            if self.tile[x, y] != 1:
                continue

            neighbours = 0
            # if north/south is walls but east/west is not this this is a candidate
            north_south = int(self.tile[x, y - 1] == 1) + int(self.tile[x, y + 1] == 1)
            east_west = int(self.tile[x - 1, y] == 1) + int(self.tile[x + 1, y] == 1)
            if (north_south == 2 and east_west == 0) or (north_south == 0 and east_west == 2):
                if np.random.rand() < 0.5:
                    self.tile[x, y] = 0

    def plot(self):
        """ plot the maze. """
        fig, ax = plt.subplots()
        ax.imshow(self.tile)
        x, y = self.state
        if x >= 0:
            c = plt.Circle((y, x), 0.2, color='red')
            ax.add_artist(c)
        plt.show()


class Maze_MDP(Maze_Base):
    """ MDP version of maze.
        Observation is location in maze
    """
    def __init__(self, width, height):
        super().__init__(width, height)
        self.observation_space = spaces.Box(np.array([0,0]),np.array([width,height]),dtype=np.float32)

    def get_observation(self):
        return self.state


class Maze_POMDP(Maze_Base):
    """ POMDP version of maze.
        Observation is if a wall exists in each of the 4 cardinal directions.
    """

    def __init__(self, width, height):
        super().__init__(width, height)
        self.observation_space = spaces.Discrete(16)

    def get_observation(self):
        """ Return a POMDP observation"""
        x,y = self.state
        bits = [self.tile[x+dx, y+dy] == 1 for (dx,dy) in self.DELTAS]
        return bits_to_int(bits)