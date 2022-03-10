import yaml, pytest, os, sys
import numpy as np
import torch as th

# Correction of PYTOHNPATH for relative imports
sys.path.append(os.path.join(os.getcwd(), 'src'))
from envs.hunt_trip import HuntingTrip
from .test_utils import float_comprasion

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
    n_agents, n_preys = env.n_actors
    assert env.grid[:, :, 0].sum() == np.arange(n_agents + 1).sum()
    assert env.grid[:, :, 1].sum() == n_preys
    for i in range(2):
        assert (env.grid[env.actors[i, :env.n_actors[i], 0], env.actors[i, :env.n_actors[i], 1], i] != 0).all()

    if env.obstacles.size:
        assert (env.grid[env.obstacles[:, 0], env.obstacles[:, 1], 2] == -1).all()
    else:
        assert (env.grid[:, :, 2] == 0).all()


def test_step():
    args = config_args.copy()
    args['map'] = "Test_Map1"
    env = HuntingTrip(**args)

    valid_actions = [arr.tolist() for arr in env.get_avail_actions()]
    assert valid_actions[0] == [0, 1, 0, 0, 1, 1] and valid_actions[1] == [0, 0, 0, 1, 1, 1] and \
        valid_actions[2] == [0, 0, 1, 1, 1, 1] and valid_actions[3] == [1, 1, 1, 1, 1, 1]
        
    actions = th.tensor([1, 5, 2, 2])
    reward, terminated, _ = env.step(actions)

    # Test step results
    assert (env.actors[0, :env.n_actors[0]] == np.asarray([[1, 0], [4, 4], [4, 2], [2, 1]])).all()
    assert float_comprasion(reward, 3 * env.reward_move + 1 * env.reward_catch)
    assert not terminated

    actions = th.tensor([4, 4, 5, 4])
    reward, _, _ = env.step(actions)
    assert float_comprasion(reward, 3 * env.reward_stay + 1 * env.reward_catch + 1 * env.reward_hunt)
    assert env.grid[:, :, 1].sum() == env.n_actors[1] - 1

    actions = th.tensor([4, 4, 3, 4])
    reward, _, _ = env.step(actions)
    assert float_comprasion(reward, 3 * env.reward_stay + 1 * env.reward_move + 1 * env.reward_carry)


def test_obs():
    args = config_args.copy()
    
    # State observation
    env = HuntingTrip(**args)
    s = env.get_state().reshape(env.height, env.width, -1)
    assert np.sum(s[:, :, 0]) == np.arange(env.n_actors[0] + 1).sum()

    # Agents observation
    os = env.get_obs()
    os = [o.reshape(env.height, env.width, -1) for o in os]
    assert all([(np.sum(o[:, :, 0] != 0) == 1) for o in os]) # only the agent cell is marked in the one-hot

    o_vector = [os[agent][env.actors[0, agent, 0], env.actors[0, agent, 1], :] for agent in range(env.n_actors[0])]
    assert all([v[0] == v[1] for v in o_vector]) # agent cell is in the right place


def test_reset():
    args = config_args.copy()
    args["episode_limit"] = 250
    env = HuntingTrip(**args)

    terminal_state = False
    while not terminal_state:
        actions = env.get_avail_actions()
        actions = th.tensor([np.random.choice(env.n_actions, p=a/sum(a)) for a in actions])
        _, terminal_state, _ = env.step(actions)
    
    env.reset()

    # Reset of coverage status
    assert env.prey_available.all()
    for i in range(2):
        assert (env.grid[env.actors[i, :env.n_actors[i], 0], env.actors[i, :env.n_actors[i], 1], i] != 0).all()

def test_e2e(rounds=100):
    args = config_args.copy()

    for i in range(rounds):
        # Set seeds for debugging and reproducability
        args['random_seed'] = i
        np.random.seed = i

        # Select random values for simulator
        args['world_shape'] = np.random.randint(4, 12,size=(2))
        args['n_agents'] = np.random.randint(2, 5)
        args['obstacle_rate'] = np.random.rand() * 0.4
        args['episode_limit'] = np.random.randint(100, 1000)

        print(f'Testing: run number #{i} with customized seed = {i}')

        env = HuntingTrip(**args)
        terminal_state = False
        while not terminal_state:
            actions = env.get_avail_actions()
            actions = th.tensor([np.random.choice(env.n_actions, p=a/sum(a)) for a in actions])
            _, terminal_state, _ = env.step(actions)
        
        assert np.sum(env.grid[:, :, 1]) == 0 or env.steps == env.episode_limit

if __name__ == '__main__':
    test_e2e()