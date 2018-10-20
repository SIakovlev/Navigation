# Navigation
Deep Reinforcement Learning Nanodegree Project 1

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

All project files are stored in `/src` folder. All project settings are stored in JSON file: `settings.json`.

#### Environment setup

- For detailed Python environment setup (PyTorch, the ML-Agents toolkit, and a few more Python packages) please follow these steps: [link](https://github.com/udacity/deep-reinforcement-learning#dependencies)

- Download pre-built Unity Environment:
  - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  - [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  - [Win x32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  - [Win x64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

 - Open `settings.json` and specify the relative path to the application file in `"path"` inside of `"env_params"`.
  
#### Settings

- `settings.json` is divided into 4 sections: 
  - `general_params` - general, module agnostic parameters: mode (`train` or `test`), number of episodes, seed.
  - `agent_params` - agent parameters: epsilon, gamma, learning rate, etc. This section also includes neural network configuration settings and memory replay buffer parameters.
  - `trainer_params` - trainer parameters: epsilon decay, learning rate decay, etc. They are responsible for any change of agent learning parameters. Agent can't change them.
  - `env_params` - environment parameters: path, number of agents, etc.

### Implementation details


#### DQN


#### Double DQN (DDQN)


#### Dueling DDNQ


#### Prioritised Replay
