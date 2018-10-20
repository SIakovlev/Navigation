# Navigation
Deep Reinforcement Learning Nanodegree Project 1

![Navigation](https://github.com/SIakovlev/Navigation/blob/master/results/navigation_short.gif)

### Project description

For this project, the task is to train an agent to navigate in a large, square world, while collecting yellow bananas, and avoiding blue bananas. A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. Thus, the goal is to collect as many yellow bananas as possible while avoiding blue bananas.

- **State space** is `37` dimensional and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. 

- **Action space** is `4` dimentional. Four discrete actions correspond to:
  - `0` - move forward
  - `1` - move backward
  - `2` - move left
  - `3` - move right

- **Solution criteria**: the environment is considered as solved when the agent gets an average score of **+13 over 100 consecutive episodes**.

### Getting started

#### Configuration

Configuration I used for this project:

- OS: Mac OS 10.14 Mojave
- i7-8800H, 32GB, Radeon Pro 560X 4GB

#### Structure

All project files are stored in `/src` folder:
- `main.py` - main file where the program execution starts.
- `agent.py` - agent class implementation.
- `unity_env.py` - Unity Environment wrapper (borrowed from [here](https://github.com/Unity-Technologies/ml-agents/blob/master/gym-unity/gym_unity/envs/unity_env.py) and modified).
- `trainer.py` - trainer (interface between agent and environment) implementation.
- `replay_buffer.py` - memory replay buffer implementation.
- `sum_tree.py` - sum tree implementation for memory replay buffer.
- `q_network.py` - neural network implementations (PyTorch)

All project settings are stored in JSON file: `settings.json`. It is divided into 4 sections: 
- `general_params` - general, module agnostic parameters: mode (`train` or `test`), number of episodes, seed.
- `agent_params` - agent parameters: epsilon, gamma, learning rate, etc. This section also includes neural network configuration settings and memory replay buffer parameters.
- `trainer_params` - trainer parameters: epsilon decay, learning rate decay, etc. They are responsible for any change of agent learning parameters. Agent can't change them.
- `env_params` - environment parameters: path, number of agents, etc.

#### Environment setup

- For detailed Python environment setup (PyTorch, the ML-Agents toolkit, and a few more Python packages) please follow these steps: [link](https://github.com/udacity/deep-reinforcement-learning#dependencies)

- Download pre-built Unity Environment:
  - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  - [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  - [Win x32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  - [Win x64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

 - Open `settings.json` and specify the relative path to the application file in `"path"` inside of `"env_params"`.

### Implementation details

The following four algorithms were implemented and tested on this environment.

#### DQN

**Idea**. Use neural network for Q-value function approximation as `state` -> `action` mapping with the following loss function minimised:

![equation](http://latex.codecogs.com/gif.latex?MSE%28r_%7Bt&plus;1%7D&plus;%5Cgamma%20%5Cmax_%7Ba%7DQ%5Et%28s_%7Bt&plus;1%7D%2C%20a%29-Q%28s_%7Bt%7D%2C%20a_%7Bt%7D%29%29)

Neural network architecture:

| Layer   | (in, out)          | Activation|
|---------|--------------------|-----------|
| Layer 1 | (`state_size`, 64) | `relu`|
| Layer 2 | (64, 128) | `relu` |
| Layer 3 | (128, 128)| `relu` |
| Layer 4 | (128, 32) | `relu` |
| Layer 5 | (32, `action_size`)| - |

#### Double DQN (DDQN)

**Idea**. Use neural network for Q-value function approximation as `state` -> `action` mapping with the following loss function minimised:

![equation](http://latex.codecogs.com/gif.latex?MSE%28r_%7Bt&plus;1%7D&plus;%5Cgamma%20Q%5Et%28s_%7Bt&plus;1%7D%2C%20%5Carg%5Cmax_%7Ba%7DQ%28s_%7Bt&plus;1%7D%2C%20a%29%29-Q%28s_%7Bt%7D%2C%20a_%7Bt%7D%29%29)

Implementation of both DQN and DDQN can be found in file `agent.py`:

```python
  loss_fn = nn.MSELoss(reduce=False)

  if self.__learning_mode['DQN']:
      Q_target_next = self.__qnetwork_target.forward(next_states).max(1)[0].unsqueeze(1).detach()
  else:
      Q_target_next = self.__qnetwork_target.forward(next_states). \
          gather(1, self.__qnetwork_local.forward(next_states).max(1)[1].unsqueeze(1)).detach()

  targets = rewards + self.gamma * Q_target_next * (1 - dones)
  outputs = self.__qnetwork_local.forward(states).gather(1, actions)
  loss = loss_fn(outputs, targets)
```

#### Dueling DDNQ

**Idea**. Use neural network for Q-value function approximation. However in this case, the neural net generates both `V(s)` - value function and `A(s, a)` - advantage function. The last layer then aggregates them together and forms `Q(s, a)`.

The dueling architecture is implemented in `q_network.py` and aggregation step is shown below: 

```python
  ...
  value = self.V(action)
  af = self.AL(action)
  action = value + (af - torch.mean(af))
  return action
```

Based on experimental evidence form the original paper ([link](https://arxiv.org/abs/1511.06581)), this should improve stability and learning rate of agents. 

#### Prioritised Replay

**Idea**. For memory replay, the agent collects tuples of `(state, reward, next_state, action, done)` and reuses them for future learning. In case of prioritised replay the agent has to assign priority to each tuple, corresponding to their contribution to learning. After that, these tuples are reused based on their priorities, leading to more efficient learning.

The most challenging part here is the data structure that is fast for search and sampling. The solution was found [here](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/) where a sum tree was suggested as a good choice. It allows accessing and sampling data in logarithmic time `O(log N)`. The implementation was taken from the same source and adjusted for this particular use case (see `sum_tree.py`).

#### Additional modifications

One more modification that seemed to work very well was slowly reducing learning rate during agent training. It lead to a higher average score (see next section where results are discussed). To control learning rate decay, one more parameter `learning_rate_decay` was introduced (see `settings.json`).

### Results

#### Algorithms comparison

The follwing graph demonstrates the average reward (over 100 episodes) for different algorithms discussed above (and their combinations):

![Comparison](https://github.com/SIakovlev/Navigation/blob/master/results/comparison.png)

The parameters for these runs remained the same and can be found in `settings.json`.

Several observations can be made:

- Prioritised replay buffer does not affect algorithm performance for this specific environment. Moreover, in some runs I have observed even worse performance (by less than 10%). A replay buffer with uniform choice seems to be enough for this task.
- Double DDQN allows to get the fastest reward collection and get 13 points (black solid line on the graph) in about 200-250 episodes. This can also be tuned by adjusting the values of `epsilon`, `tau` (target network update time constant) and `learning rate`).
- All algorithms allow getting over 16 points on average in less than 1800 episodes. This satisfied the solution criteria. 

Another interesting effect can be seen from the graph below:

![The effect of decreasing learning rate](https://github.com/SIakovlev/Navigation/blob/master/results/lr_decreasing_effect.png)

I have made two separate runs with the same parameters but in the first case I kept learning rate fixed, whereas in the second case I decreased it after each episode by about 20%. This graph clearly shows how decreasing the learning rate helps agent to get more reward. The possible explanation is that lowering learning rate leads to smaller changes in Q network at later steps which helps convergence.

#### Visualisation of agent's behaviour

The following gif file shows one agent run where it manages to collect 20 bananas:

![Navigation](https://github.com/SIakovlev/Navigation/blob/master/results/navigation.gif)
