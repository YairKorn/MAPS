from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np


class StagHunt(MultiAgentEnv):

    action_labels = {'stay': 0, 'right': 1, 'down': 2, 'left': 3, 'up': 4}

    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        # Observation properties
        #? self.truncate_episodes = getattr(args, "truncate_episodes", True)
        self.observe_ids = getattr(args, "observe_ids", False)
        self.watch_covered = getattr(args, "watch_covered", False)
        self.watch_surface = getattr(args, "watch_surface", False)
        self.n_feats = 1 + self.watch_covered + self.watch_surface # features per cell

        # Build the environment
        world_shape = np.asarray(args.world_shape, dtype=np.int16) 
        self.height, self.width = world_shape
        self.toroidal = args.toroidal

        # Action space
        self.n_actions     = 5 
        self.action_effect = np.asarray([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int16)

        # Initialization
        self.grid = np.zeros((self.batch_size, self.x_max, self.y_max, self.n_feats), dtype=np.float16)
        
        obstacles = getattr(args, "obstacles_location", []) # locating obstacles on the map
        self.grid[obstacles[:, 0], obstacles[:, 1]] = -1
        
        threats   = getattr(args, "threat_location",    []) # locating threats on the map
        self.grid[threats[:, 0], threats[:, 1]] = threats[:, 2]

        self.episode_limit = args.episode_limit

        # Reward function
        self.time_reward      = getattr(args, "reward_time", -0.1)
        self.collision_reward = getattr(args, "reward_collision", 0.0)
        self.new_cell_reward  = getattr(args, "reward_cell", 1.0)
        self.succes_reward    = getattr(args, "succes_reward", 0.0)
        self.threat_reward    = getattr(args, "threat_reward", 0.0) # this constant is multiplied by the designed reward function

        self.simulated_mode   = getattr(args, "simulated_mode", False) # never disable a robot but give a negative reward for entering a threat