import torch
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np

from buffer import UniformReplayBuffer
from model import OriginalDQN, DuelDQN

LR = 5e-4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64

GAMMA = .99

TAU = 1e-3
UPDATE_EVERY = 4

class DQNAgent:
    def __init__(self, state_size, action_size, double_dqn, dueling_dqn, seed):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.double_dqn = double_dqn
        self.dueling = dueling_dqn
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        if self.dueling:
            self.qnetwork_local = DuelDQN(self.state_size, self.action_size, seed).to(self.device)
            self.qnetwork_target = DuelDQN(self.state_size, self.action_size, seed).to(self.device)
        else:
            self.qnetwork_local = OriginalDQN(self.state_size, self.action_size, seed).to(self.device)
            self.qnetwork_target = OriginalDQN(self.state_size, self.action_size, seed).to(self.device)
            
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        self.replay_buffer = UniformReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed, self.device)
        
        self.t_step = 0
        
        print('Device used: {}'.format(self.device)) 
        print('Double DQN?', self.double_dqn)
        print('Dueling?', self.dueling)
        print('Local DQN ->', self.qnetwork_local)
        print('Target DQN ->', self.qnetwork_target)    
    
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