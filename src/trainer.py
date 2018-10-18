import gym
import random
import numpy as np
import time
import sys
import torch
from dqn_agent import Agent
from unity_env import UnityEnv
from collections import deque
import pickle
import datetime
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='../logs/trainer' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M') + '.log', level=logging.INFO)

class Trainer:
    def __init__(self):
        self.env = UnityEnv('../BananaApp/Banana.app')

        #self.env = gym.make('LunarLander-v2')
        #self.env = gym.make('CartPole-v0')
        self.env.seed(0)

        self.agent = Agent(state_size=self.env.observation_space.shape[0],
                           action_size=self.env.action_space.n, seed=0, learning_mode='DQN')

        # training parameters
        self.max_eps = 1.0
        self.final_eps = 0.01
        self.eps_decay = 0.98
        self.agent_eps = 1.0
        self.agent_learning_rate = 0.001

        # increase parameter beta
        self.agent_beta = 0.0
        self.agent_alpha = 0.0

        # training data gathering
        self.avg_rewards = []
        self.scores = []
        self.score = 0

    def train(self, num_of_episodes=2000, max_t=1000):
        reward_window = deque(maxlen=100)
        self.agent_eps = 1.0
        #self.eps_decay = np.exp(np.log(self.final_eps / self.max_eps) / (0.8 * num_of_episodes))
        prev_total_reward = 0
        reward_matrix = np.zeros((num_of_episodes, 300))

        for episode_i in range(num_of_episodes):

            state = self.env.reset()
            done = False
            total_reward = 0
            total_loss = 0

            self.agent_eps = self.max_eps/(episode_i + 1)
            #self.agent_eps *= self.eps_decay
            self.agent.set_eps(self.agent_eps)

            self.agent_beta = 1 - np.exp(-0.003 * episode_i)
            self.agent.set_beta(self.agent_beta)

            counter = 0
            #for t in range(max_t):
            while not done:
                action, rn = self.agent.choose_action(state)
                #action = random.randint(0, 3)

                next_state, reward, done, _ = self.env.step(action)
                logging.info("epsiode: {}, reward: {}, counter: {}, action: {}, actions: {}".format(episode_i, reward, counter, action, rn))
                # if reward:
                #     print("reward: {}, counter: {}, action: {}".format(reward, counter, action))
                total_reward += reward

                reward_matrix[episode_i, counter] = reward

                self.agent.step(state, action, reward, next_state, done)

                total_loss += self.agent.agent_loss

                state = next_state

                counter += 1

            reward_window.append(total_reward)

            print('\rEpisode {}\tCurrent Score: {:.2f}\tAverage Score: {:.2f} \t\tTotal loss: {:.2f}\tEpsilon: {:.2f}\tBeta: {:.2f}\tLearning rate: {:.4f}'.
                  format(episode_i, total_reward, np.mean(reward_window), total_loss, self.agent_eps, self.agent_beta, self.agent_learning_rate), end="")
            if episode_i % 100 == 0:

                self.agent_learning_rate /= 1.2
                self.agent.set_learning_rate(self.agent_learning_rate)

                avg_reward = np.mean(np.array(reward_window))
                print("\rEpisode: {}\tAverage total reward: {:.2f}".format(episode_i, avg_reward))
                self.avg_rewards.append(avg_reward)

                if avg_reward >= 15.0:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_i - 100,
                                                                                                 avg_reward))
                    torch.save(self.agent.get_qlocal().state_dict(), '../results/checkpoint_new.pth')

        t = datetime.datetime.now()

        reward_matrix.dump('../results/reward_matrix_new_{}.dat'.format(str(t.hour) + '_' + str(t.minute)))
        np.array(self.avg_rewards).dump('../results/average_rewards_new_{}.dat'.format(str(t.hour) + '_' + str(t.minute)))

    def test(self, time_span=10, checkpoint_path='../results/checkpoint.pth'):

        self.agent.get_qlocal().load_state_dict(torch.load(checkpoint_path))

        print("1st layer's weights: {}".format(self.agent.get_qlocal().fc1.weight.data))
        print("2nd layer's weights: {}".format(self.agent.get_qlocal().fc2.weight.data))
        print("3rd layer's weights: {}".format(self.agent.get_qlocal().fc3.weight.data))
        print("4th layer's weights: {}".format(self.agent.get_qlocal().fc4.weight.data))

        for t in range(time_span):
            state = self.env.reset(train_mode=False)
            self.score = 0
            done = False

            while not done:
                action = self.agent.choose_action(state, 'test')
                sys.stdout.flush()
                self.env.render()
                state, reward, done, _ = self.env.step(action)
                self.score += reward

            print('\nFinal score:', self.score)

        self.env.close()
