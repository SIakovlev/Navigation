from pprint import pprint
import json
import trainer
import matplotlib as mpl
# mpl.use('TkAgg')  # Mac OS specific


def main(**kwargs):
    # Parse JSON settings file
    general_params = kwargs['general_params']

    num_of_episodes = general_params['num_of_episodes']
    mode = general_params['mode']

    t = trainer.Trainer(kwargs)
    if mode['train']:
        t.train(num_of_episodes=num_of_episodes)
    else:
        t.test(checkpoint_filename='checkpoint.pth', time_span=3)


if __name__ == '__main__':

    with open('../settings.json') as settings:
        params = json.load(settings)

    main(**params)
