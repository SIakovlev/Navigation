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

LR = 0.001
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 100
UPDATE_EVERY = 4
TAU = 1e-3
GAMMA = 0.99
E = 1e-6

class Agent():
    def __init__(self, action_size, state_size, seed, learning_mode='DQN'):
        self.__action_size = action_size
        self.__state_size = state_size
        #random.seed(seed)

        self.__qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.__qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.__optimiser = optim.Adam(self.__qnetwork_local.parameters(), lr=LR)

        self.__memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.__t = 0
        self.agent_loss = 0

        # intrinsic parameters
        self.__learning_mode = learning_mode
        self.__eps = 1.0
        self.__gamma = GAMMA
        self.__beta = 0.0
        self.__alpha = 0.0
        self.__learning_rate = 0.001

    def step(self, state, action, reward, next_state, done):
        # add experience to memory
        self.__memory.add(state, action, reward, next_state, done)

        self.__t = (self.__t + 1) % UPDATE_EVERY
        if not self.__t:
            if len(self.__memory) >= BATCH_SIZE:
                experiences = self.__memory.sample()
                self.__update(experiences)

    def set_eps(self, eps):
        self.__eps = eps

    def set_beta(self, beta):
        self.__beta = beta

    def set_alpha(self, alpha):
        self.__beta = alpha

    def set_learning_rate(self, lr):
        self.__learning_rate = lr
        for param_group in self.__optimiser.param_groups:
            param_group['lr'] = lr

    def get_qlocal(self):
        return self.__qnetwork_local

    def choose_action(self, state, mode='train'):
        # state should be transformed to a tensor
        if mode == 'train':
            random_number = random.random()
            if random_number > self.__eps:
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                #self.__qnetwork_local.eval()
                with torch.no_grad():
                    actions = self.__qnetwork_local(state)
                #self.__qnetwork_local.train()
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


    def __update(self, experiences):
        states, actions, rewards, next_states, dones, indices, probs = experiences
        # Compute and minimise the loss
        self.__optimiser.zero_grad()

        loss_fn = nn.MSELoss(reduce=False)

        # Calculate weights and normalise
        weights = [(prob * len(self.__memory)) ** (-self.__beta) for prob in probs]
        weights = np.array([w / max(weights) for w in weights]).reshape((-1, 1))

        if self.__learning_mode == 'DQN':
            Q_target_next = self.__qnetwork_target.forward(next_states).max(1)[0].unsqueeze(1).detach()
        else:
            Q_target_next = self.__qnetwork_target.forward(next_states). \
                gather(1, self.__qnetwork_local.forward(next_states).max(1)[1].unsqueeze(1)).detach()

        targets = rewards + self.__gamma * Q_target_next * (1 - dones)
        outputs = self.__qnetwork_local.forward(states).gather(1, actions)
        loss = loss_fn(outputs, targets)

        # Calculate weighted loss
        weighted_loss = torch.mean(torch.from_numpy(weights).float() * loss)
        weighted_loss.backward()

        self.__optimiser.step()

        self.__memory.update(indices, list(loss.detach().numpy().squeeze() ** self.__alpha + E))
        self.__soft_update(self.__qnetwork_local, self.__qnetwork_target, TAU)

        self.agent_loss = weighted_loss.detach().numpy().squeeze()

    def __soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)