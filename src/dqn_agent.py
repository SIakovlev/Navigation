import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn

#random.seed(0)
#torch.manual_seed(0)

from q_network import QNetwork, DuelingQNetwork
from replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, params):
        action_size = params['action_size']
        state_size = params['state_size']
        buf_params = params['buf_params']
        nn_params = params['nn_params']
        nn_params['l1'][0] = state_size
        nn_params['l5'][1] = action_size

        self.__learning_mode = params['learning_mode']

        if self.__learning_mode['DuelingDDQN']:
            self.__qnetwork_local = DuelingQNetwork(nn_params).to(device)
            self.__qnetwork_target = DuelingQNetwork(nn_params).to(device)
        else:
            self.__qnetwork_local = QNetwork(nn_params).to(device)
            self.__qnetwork_target = QNetwork(nn_params).to(device)

        self.__action_size = action_size
        self.__state_size = state_size
        self.__memory = ReplayBuffer(buf_params)
        self.__t = 0

        self.eps = params['eps_initial']
        self.gamma = params['gamma']
        self.learning_rate = params['learning_rate']
        self.update_period = params['update_period']
        self.a = params['a']
        self.b = params['b']
        self.e = params['e']
        self.tau = params['tau']

        self.__optimiser = optim.Adam(self.__qnetwork_local.parameters(), self.learning_rate)

        # other parameters
        self.agent_loss = 0.0

    # Set methods
    def set_learning_rate(self, lr):
        self.learning_rate = lr
        for param_group in self.__optimiser.param_groups:
            param_group['lr'] = lr

    # Get methods
    def get_qlocal(self):
        return self.__qnetwork_local

    # Other methods
    def step(self, state, action, reward, next_state, done):
        # add experience to memory
        self.__memory.add(state, action, reward, next_state, done)

        self.__t = (self.__t + 1) % self.update_period
        if not self.__t:
            if self.__memory.is_ready():
                experiences = self.__memory.sample()
                self.__update(experiences, self.b, self.a, self.e, self.tau)

    def choose_action(self, state, mode='train'):
        # state should be transformed to a tensor
        if mode == 'train':
            if random.random() > self.eps:
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                self.__qnetwork_local.eval()
                with torch.no_grad():
                    actions = self.__qnetwork_local(state)
                self.__qnetwork_local.train()
                return np.argmax(actions.cpu().data.numpy()), actions
            else:
                return np.random.choice(np.arange(self.__action_size)), 0
        elif mode == 'test':
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.__qnetwork_local.eval()
            with torch.no_grad():
                actions = self.__qnetwork_local(state)
            self.__qnetwork_local.train()
            return np.argmax(actions.cpu().data.numpy())
        else:
            print("Invalid mode value")

    def __update(self, experiences, beta, alpha, e, tau):
        states, actions, rewards, next_states, dones, indices, probs = experiences
        # Compute and minimise the loss
        self.__optimiser.zero_grad()

        loss_fn = nn.MSELoss(reduce=False)

        if self.__learning_mode['DQN']:
            Q_target_next = self.__qnetwork_target.forward(next_states).max(1)[0].unsqueeze(1).detach()
        else:
            Q_target_next = self.__qnetwork_target.forward(next_states). \
                gather(1, self.__qnetwork_local.forward(next_states).max(1)[1].unsqueeze(1)).detach()

        targets = rewards + self.gamma * Q_target_next * (1 - dones)
        outputs = self.__qnetwork_local.forward(states).gather(1, actions)
        loss = loss_fn(outputs, targets)

        # Calculate weights and normalise
        if probs:
            weights = [(prob * len(self.__memory)) ** (-beta) for prob in probs]
            weights = np.array([w / max(weights) for w in weights]).reshape((-1, 1))
        else:
            weights = np.ones(loss.shape, dtype=np.float)

        if indices:
            self.__memory.update(indices, list(loss.detach().numpy().squeeze() ** alpha + e))

        # Calculate weighted loss
        weighted_loss = torch.mean(torch.from_numpy(weights).float() * loss)
        weighted_loss.backward()

        self.__optimiser.step()

        self.__soft_update(self.__qnetwork_local, self.__qnetwork_target, tau)

        self.agent_loss = weighted_loss.detach().numpy().squeeze()

    def __soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)