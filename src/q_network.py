import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):


        # the network was: s - 64|64 - 128|128 - 64|64 - a
        # the network is: s - 64|64 - 96|96 - 96|96 - 64|64 - a
        super(QNetwork, self).__init__()
        #torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        #nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(64, 96)
        #nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(96, 96)
        #nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(96, 64)
        #nn.init.xavier_uniform_(self.fc4.weight)
        self.fc5 = nn.Linear(64, action_size)
        #nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, state):

        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        action = F.relu(self.fc3(action))
        action = F.relu(self.fc4(action))
        action = self.fc5(action)
        return action


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(64, 96)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(96, 96)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(96, 64)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.V = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.V.weight)
        self.AL = nn.Linear(64, action_size)
        nn.init.xavier_uniform_(self.AL.weight)

    def forward(self, state):
        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        action = F.relu(self.fc3(action))
        action = F.relu(self.fc4(action))
        value = self.V(action)
        af = self.AL(action)
        action = value + (af - torch.mean(af))
        return action


class NoisyQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        super(NoisyQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(64, 96)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(96, 96)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(96, 64)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.V = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.V.weight)
        self.AL = nn.Linear(64, action_size)
        nn.init.xavier_uniform_(self.AL.weight)

    def forward(self, state):
        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        action = F.relu(self.fc3(action))
        action = F.relu(self.fc4(action))
        value = self.V(action)
        af = self.AL(action)
        action = value + (af - torch.mean(af))
        return action