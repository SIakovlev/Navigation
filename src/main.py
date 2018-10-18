import random
import torch
import numpy as np

import trainer
import matplotlib as mpl
mpl.use('TkAgg')  # Mac OS specific


def main():

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.set_default_tensor_type('torch.FloatTensor')

    t = trainer.Trainer()
    t.train(50, 1000)
    #t.test(3, checkpoint_path='../results/checkpoint_dqn.pth')


if __name__ == '__main__':
    main()
