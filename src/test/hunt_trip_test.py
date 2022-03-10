import yaml, pytest, os, sys
import numpy as np
import torch as th

# Correction of PYTOHNPATH for relative imports
sys.path.append(os.path.join(os.getcwd(), 'src'))
from envs.hunt_trip import HuntingTrip
from test_utils import float_comprasion

# TODO: Further test: 
# - disabling of a robot

# args are reseted every test
CONFIG_PATH = os.path.join('.', 'src', 'test', 'hunt_trip_test.yaml')
with open(CONFIG_PATH, 'r') as config:
    config_args = yaml.safe_load(config)['env_args']


def test_init():
    args = config_args.copy()
    # Initialization without error, should pass
    env = HuntingTrip(**args)

    # Succesfully locating agents & preys
    assert env.grid[:, :, 0] == np.arange(args['n_agents']).sum()
    assert env.grid[:, :, 1] == args['n_preys']
    for i in range(2):
        assert (env.grid[env.actors[i, :, 0], env.actors[i, :, 1], i] != 0).all()
    assert (env.grid[env.obstacles[:, 0], env.obstacles[:, 1], 2] == -1).all()


def test_step():
    pass


if __name__ == '__main__':
    test_init()