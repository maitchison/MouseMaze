"""
Collection of useful functions for reinforcement learning.
"""

import numpy as np

def show_moves(agent, env):
    """ Shows some examples moves between agent and enviroment. """
    
    print("Playing game with agent {} in environment {}".format(agent, env)) 
    obs, reward = env.reset(), 0    
    for i in range(10):
        action = agent.act(obs,reward)
        print("Saw observation {} with reward {} and played action {}".format(obs, reward, action))
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break        
            
def evaluate_agent(agent, env, max_steps = 1000):
    """ Evaluates the agents performance on the environment. 
        @returns Reward history at every step.
    """
        
    obs, reward = env.reset(), 0    
    total_reward = 0
    reward_history = []
    
    for i in range(max_steps):
        action = agent.act(obs,reward)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        reward_history.append(total_reward)
        if done:
            break 
        
    return reward_history
        
def plot_reward_history(agent, env):
    
    steps = 1000
    trials = 100
    
    total_reward = np.zeros(steps)

    for i in range(trials):
        total_reward += np.array(evaluate_agent(agent, env, steps)) / trials

    plt.plot(range(steps), total_reward, label=agent)        
    plt.xlabel("Step")
    plt.ylabel("Average Reward")    
    plt.legend()
    plt.show()