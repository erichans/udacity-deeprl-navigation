from buffer import UniformReplayBuffer
import random
from collections import deque

#!/usr/bin/env python
# coding: utf-8

# In[1]:


from unityagents import UnityEnvironment
import numpy as np


# In[2]:


env = UnityEnvironment(file_name="Banana_Windows_x86_64/banana.exe")


# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# In[4]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


# In[5]:

# In[6]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, dueling, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = dueling
        
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        
        if self.dueling:
            self.fc_advantage = nn.Linear(32, action_size)
            self.fc_value = nn.Linear(32, 1)
        else:
            self.fc4 = nn.Linear(32, action_size)
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        if self.dueling:
            value = self.fc_value(x)
            advantage = self.fc_advantage(x)
            return value + advantage - advantage.mean()
        else:
            return self.fc4(x)
    


# In[7]:


import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = .99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

class VanillaDQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('Device used: {}'.format(self.device)) 
#         self.device = 'cpu'
        self.double_dqn = False
        self.dueling = True
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.dueling, seed).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.dueling, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        self.replay_buffer = UniformReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device)
        
        self.t_step = 0
    
    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(state).float().to(self.device)
        
        if random.random() > epsilon:
            self.qnetwork_local.eval()
            with torch.no_grad():
                action = np.argmax(self.qnetwork_local(state).cpu().data.numpy())
            self.qnetwork_local.train()
        else:
            action = np.random.randint(self.action_size)
        
        return int(action)
    
    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.replay_buffer) >= BATCH_SIZE:
            self._learn(self.replay_buffer.sample(), GAMMA)
    
    def _learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        if self.double_dqn:
            Q_action_targets_next = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_action_targets_next)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        
        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_expected, Q_target)
        loss.backward()
        self.optimizer.step()
        
        self._soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
    
    def _soft_update(self, local_model, target_model, tau):
        for local_parameter, target_parameter in zip(local_model.parameters(), target_model.parameters()):
            target_parameter.data.copy_((1.0-tau)*target_parameter+(tau*local_parameter))


# In[8]:


# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

agent = VanillaDQNAgent(state_size, action_size, 42)

TOTAL_EPISODES = 500
EPSILON_START = 1.0
EPSILON_DECAY = .99
EPSILON_END = 0.01

def dqn(episodes=2500, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for episode in range(1, episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        score = 0.0
        done = False
        while not done:
            state = env_info.vector_observations[0]
            action = agent.act(state, epsilon=eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward

        scores_window.append(score)
        scores.append(score)
        eps = max(eps*eps_decay, eps_end)
        print('\rEpisode {}/{}\tAverage Score: {:.2f}'.format(episode, TOTAL_EPISODES, np.mean(scores_window)), end='')
        if episode % 100 == 0:
            print('\rEpisode {}/{}\tAverage Score: {:.2f}'.format(episode, TOTAL_EPISODES, np.mean(scores_window)))
        if np.mean(scores_window) > 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    return scores
        
scores = dqn(TOTAL_EPISODES, EPSILON_START, EPSILON_END, EPSILON_DECAY)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.xlabel('Episode #')
# plt.ylabel('Score')
# plt.show()


# In[9]:


values = []
eps = EPSILON_START
for i in range(len(scores)-100):
    values.append(eps)
    eps = max(eps*EPSILON_DECAY, EPSILON_END)

print('Last 100 Epsilon', values[-100])
#plt.plot(np.arange(len(values)), values)
#cpu: 276, 286, 349
#gpu: 324, 311


# In[10]:


env.close()

