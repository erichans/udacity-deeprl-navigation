from agent import DQNAgent

from collections import deque

from unityagents import UnityEnvironment

import torch
import numpy as np

import matplotlib.pyplot as plt

def start_env():
    env = UnityEnvironment(file_name="Banana_Windows_x86_64/banana.exe")

    # get the default brain
    brain_name = get_brain_name(env)
    brain = get_brain(env)

    env_info = reset_env_info(env)

    print('Number of agents:', len(env_info.agents))

    action_size = get_action_size(env)
    print('Number of actions:', action_size)

    state = env_info.vector_observations[0]
    print('States look like:', state)
    print('States have length:', get_state_size(env_info))
    
    return env

def get_brain_name(env):
    return env.brain_names[0]
    
def get_brain(env):
    return env.brains[get_brain_name(env)]
    
def get_state_size(env_info):
    return len(env_info.vector_observations[0])
    
def get_action_size(env):
    return get_brain(env).vector_action_space_size
    
def reset_env_info(env):
    return env.reset(train_mode=True)[get_brain_name(env)]
    
def env_step(env, action):
    return env.step(action)[get_brain_name(env)]


def dqn_run(episodes=2500, eps_start=1.0, eps_end=0.01, eps_decay=0.995, double_dqn=False, dueling_dqn=False, seed=42):
    env = start_env()
    env_info = reset_env_info(env)
    
    state_size = get_state_size(env_info)
    action_size = get_action_size(env)
    
    print('Seed used:', seed)
    agent = DQNAgent(state_size, action_size, double_dqn, dueling_dqn, seed)
    
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for episode in range(1, episodes+1):
        env_info = reset_env_info(env)
        score = 0.0
        done = False
        while not done:
            state = env_info.vector_observations[0]
            action = agent.act(state, epsilon=eps)
            env_info = env_step(env, action)
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward

        scores_window.append(score)
        scores.append(score)
        eps = max(eps*eps_decay, eps_end)
        print('\rEpisode {}/{}\tAverage Score: {:.2f}, epsilon: {:.3f}'.format(episode, episodes, np.mean(scores_window), eps), end='     ')
        if episode % 100 == 0:
            print('\rEpisode {}/{}\tAverage Score: {:.2f}, epsilon: {:.3f}'.format(episode, episodes, np.mean(scores_window), eps))
        if np.mean(scores_window) > 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    
    env.close()
    return scores
    
def save_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('Episode #')
    plt.ylabel('Score')
    plt.savefig('score-evolution.png')
    #plt.show()

TOTAL_EPISODES = 500
EPSILON_START = 1.0
EPSILON_DECAY = .99
EPSILON_END = 0.01
DOUBLE_DQN = True
DUELING_DQN = True
SEED = 42

if __name__ == '__main__':    
    scores = dqn_run(TOTAL_EPISODES, EPSILON_START, EPSILON_END, EPSILON_DECAY, DOUBLE_DQN, DUELING_DQN, SEED)
    save_scores(scores)