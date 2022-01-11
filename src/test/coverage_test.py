import yaml, pytest, os
import numpy as np
from src.envs.adv_coverage import AdversarialCoverage

# TODO: Further test: 
# - reward function
# - disabling of a robot
# - configuration without obstacles
# ! update test of avail actions 

# args are reseted every test
CONFIG_PATH = os.path.join('.', 'src', 'test', 'coverage_test.yaml')
with open(CONFIG_PATH, 'r') as config:
    config_args = yaml.safe_load(config)['env_args']


def test_init():
    args = config_args.copy()
    # Initialization without error, should pass
    env = AdversarialCoverage(**args)

    # Edge cases
    with pytest.raises(ValueError, match=r"Failed .*"):
        args['random_placement'] = False # not enough agents allocations
        args['agents_placement'] = [[1, 1]]
        env = AdversarialCoverage(**args)

    with pytest.raises(ValueError, match=r"Agents .*"):
        args['random_config'] = False # agent is located on an obstacle
        args['agents_placement'] = [[1, 1], [1, 2]]
        env = AdversarialCoverage(**args)

    args['random_config'] = True
    args['random_placement'] = True

    # to many obstacles - grid is not fully-reachable, then reduce the obstacle rate
    args['obstacle_rate'] = 0.75 
    env = AdversarialCoverage(**args)
    
    # to many obstacles - not enough place for the agents, then reduce the obstacle rate
    args['obstacle_rate'] = 1.00 
    env = AdversarialCoverage(**args)
    args['obstacle_rate'] = 0.20 

    # ... Random allocation of threats
    args['threats_rate'] = 1.00
    args['risk_avg'] = 0.50
    args['risk_std'] = 1.00
    env = AdversarialCoverage(**args)
    test_grid = env.grid[:, :, 2]
    assert ((0 <= test_grid) * (test_grid <= 1) == (test_grid != -1)).all()


def test_step():
    args = config_args.copy()
    args['random_placement'] = False
    args['agents_placement'] = [[0, 1], [2, 3]]
    env = AdversarialCoverage(**args)
    
    # available actions are calculated correctly
    valid_actions = env.get_avail_actions()
    assert valid_actions[0].tolist() == [0, 4] and valid_actions[1].tolist() == [1, 2, 3, 4]
    
    actions = np.array([0, 1])
    reward, terminated, info = env.step(actions)

    #! cont: apply actions and calculate reward
    #! change rewards and recalculate


def test_observation():
    args = config_args.copy()
    
    # State observation
    env = AdversarialCoverage(**args)
    o = env.get_state()
    assert np.sum(o[::3]) == np.sum(np.arange(env.n_agents + 1))

    env.observe_ids = False # observation without agents' IDs
    o = env.get_state()
    assert np.sum(o[::3]) == env.n_agents

    # Agents observation
    os = env.get_obs()
    os = [o.reshape(env.height, env.width, -1) for o in os]
    assert all([(np.sum(o[:, :, 0] != 0) == 1) for o in os]) # only the agent cell is marked in the one-hot

    o_vector = [os[agent][env.agents[agent, 0], env.agents[agent, 1], :] for agent in range(env.n_agents)]
    assert all([v[0] == v[1] for v in o_vector]) # agent cell is in the right place


def test_reset():
    args = config_args.copy()
    env = AdversarialCoverage(**args)
    terminal_state = False
    while not terminal_state:
        actions = env.get_avail_actions()
        actions = np.asarray([np.random.choice(a) for a in actions])
        _, terminal_state, _ = env.step(actions)
    
    env.reset()

    # Reset of coverage status
    cover_status = (env.grid[:, :, 2] == -1) + (env.grid[:, :, 0] > 0)
    assert np.array_equal(env.grid[:, :, 1] == 1, (cover_status > 0))

    # All agents were revived
    assert all(env.agents_enabled)


# run the simulator end to end
def test_e2e(rounds=100):
    args = config_args.copy()
    args['shuffle_config'] = True # shuffle config for test more cases (and the full functionability)

    for i in range(rounds):
        # Set seeds for debugging and reproducability
        args['random_seed'] = i
        np.random.seed = i

        # Select random values for simulator
        args['world_shape'] = np.random.randint(4, 12,size=(2))
        args['n_agents'] = np.random.randint(2, 5)
        args['obstacle_rate'] = np.random.rand() * 0.4
        args['threats_rate'] = np.random.rand()
        args['risk_avg'] = np.random.rand() * 0.3
        args['risk_std'] = np.random.rand() * 0.4
        args['episode_limit'] = np.random.randint(100, 1000)

        print(f'Testing: run number #{i} with customized seed = {i}')

        env = AdversarialCoverage(**args)
        terminal_state = False
        while not terminal_state:
            actions = env.get_avail_actions()
            actions = np.asarray([np.random.choice(a) for a in actions])
            _, terminal_state, _ = env.step(actions)
        
        assert np.sum(env.grid[:, :, 1]) == env.n_cells or env.steps == env.episode_limit or not any(env.agents_enabled)


if __name__ == '__main__':
    test_reset()