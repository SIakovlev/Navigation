import gym
import random
import numpy as np
import sys
import torch
from agent import Agent
from unity_env import UnityEnv
from collections import deque
import datetime
import logging
from pprint import pprint

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='../logs/run_' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M') + '.log',
                    level=logging.INFO)


class Trainer:
    def __init__(self, params):

        seed = params['general_params']['seed']
        self.__set_seed(seed=seed)

        env_params = params['env_params']
        env_params['seed'] = seed
        self.env = UnityEnv(params=env_params)

        agent_params = params['agent_params']
        agent_params['state_size'] = self.env.observation_space.shape[0]
        agent_params['action_size'] = self.env.action_space.n
        self.agent = Agent(params=agent_params)

        trainer_params = params['trainer_params']
        self.learning_rate_decay = trainer_params['learning_rate_decay']
        self.max_eps = trainer_params['max_eps']
        self.final_eps = trainer_params['final_eps']
        self.eps_decay = trainer_params['eps_decay']
        self.b_decay = trainer_params['b_decay']
        self.results_path = trainer_params['results_path']
        self.model_path = trainer_params['model_path']

        # data gathering variables
        self.avg_rewards = []
        self.scores = []
        self.score = 0

        print("Configuration:")
        pprint(params)
        logging.info("Configuration: {}".format(params))

    def train(self, num_of_episodes):

        reward_window = deque(maxlen=100)

        self.eps_decay = (self.final_eps / self.max_eps) ** (1 / (0.2 * num_of_episodes))

        reward_matrix = np.zeros((num_of_episodes, 300))

        for episode_i in range(1, num_of_episodes):

            state = self.env.reset()
            done = False
            total_reward = 0
            total_loss = 0

            #self.agent.eps = self.max_eps/(episode_i + 1)
            self.agent.eps *= self.eps_decay

            #self.agent.b = 1 - np.exp(-self.b_decay * episode_i)

            counter = 0
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state

                # DEBUG
                # logging.info("epsiode: {}, reward: {}, counter: {}, action: {}, actions: {}, fc1 weight data: {}".
                #              format(episode_i, reward, counter, action, actions,
                #                     self.agent.get_qlocal().fc1.weight.data))

                total_loss += self.agent.agent_loss
                total_reward += reward
                reward_matrix[episode_i, counter] = reward
                counter += 1

            reward_window.append(total_reward)

            print('\rEpisode {}\tCurrent Score: {:.2f}\tAverage Score: {:.2f} '
                  '\t\tTotal loss: {:.2f}\tEpsilon: {:.2f}\tBeta: {:.2f}\tLearning rate: {:.4f}'.
                  format(episode_i, total_reward, np.mean(reward_window),
                         total_loss, self.agent.eps, self.agent.b, self.agent.learning_rate), end="")

            logging.info('Episode {}\tCurrent Score: {:.2f}\tAverage Score: {:.2f} '
                         '\t\tTotal loss: {:.2f}\tEpsilon: {:.2f}\tBeta: {:.2f}\tLearning rate: {:.4f}'.
                         format(episode_i, total_reward, np.mean(reward_window),
                                total_loss, self.agent.eps, self.agent.b, self.agent.learning_rate))

            self.agent.learning_rate *= self.learning_rate_decay
            self.agent.set_learning_rate(self.agent.learning_rate)

            if episode_i % 100 == 0:

                avg_reward = np.mean(np.array(reward_window))
                print("\rEpisode: {}\tAverage total reward: {:.2f}".format(episode_i, avg_reward))
                self.avg_rewards.append(avg_reward)

                if avg_reward >= 13.0:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_i - 100,
                                                                                                 avg_reward))
                    torch.save(self.agent.get_qlocal().state_dict(), self.model_path + 'checkpoint_{}.pth'.format(
                        datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')))

        t = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
        reward_matrix.dump(self.results_path + 'reward_matrix_new_{}.dat'.format(t))
        np.array(self.avg_rewards).dump(self.results_path + 'average_rewards_new_{}.dat'.format(t))

    def test(self, checkpoint_filename, time_span=10):

        checkpoint_path = self.model_path + checkpoint_filename
        self.agent.get_qlocal().load_state_dict(torch.load(checkpoint_path))
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

    @staticmethod
    def __set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)