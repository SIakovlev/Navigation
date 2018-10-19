import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, params):

        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(params['l1'][0], params['l1'][1])
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(params['l2'][0], params['l2'][1])
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(params['l3'][0], params['l3'][1])
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(params['l4'][0], params['l4'][1])
        nn.init.xavier_uniform_(self.fc4.weight)
        self.fc5 = nn.Linear(params['l5'][0], params['l5'][1])
        nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, state):

        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        action = F.relu(self.fc3(action))
        action = F.relu(self.fc4(action))
        action = self.fc5(action)
        return action


class DuelingQNetwork(nn.Module):
    """ Dueling Actor (Policy) Model."""

    def __init__(self, params):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(params['l1'][0], params['l1'][1])
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(params['l2'][0], params['l2'][1])
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(params['l3'][0], params['l3'][1])
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(params['l4'][0], params['l4'][1])
        nn.init.xavier_uniform_(self.fc4.weight)
        self.AL = nn.Linear(params['l5'][0], params['l5'][1])
        nn.init.xavier_uniform_(self.AL.weight)
        self.V = nn.Linear(params['l5'][0], 1)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, state):
        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        action = F.relu(self.fc3(action))
        action = F.relu(self.fc4(action))
        value = self.V(action)
        af = self.AL(action)
        action = value + (af - torch.mean(af))
        return action
