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
    pass


if __name__ == '__main__':
    test_init()