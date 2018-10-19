from pprint import pprint
import json
import trainer
import matplotlib as mpl
mpl.use('TkAgg')  # Mac OS specific


def main(**kwargs):
    # Parse JSON settings file
    general_params = kwargs['general_params']

    num_of_episodes = general_params['num_of_episodes']

    t = trainer.Trainer(kwargs)
    t.train(num_of_episodes=num_of_episodes)
    # t.test(3, checkpoint_filename='../results/checkpoint_dqn.pth')


if __name__ == '__main__':

    with open('../settings.json') as settings:
        params = json.load(settings)

    main(**params)
