import numpy as np
import torch as th
from .model_v1 import ActionModel

class AdvCoverage(ActionModel):
    def __init__(self, scheme, args) -> None:
        super().__init__(scheme, args)
        
        # Basics of action model
        self.action_effect = th.tensor([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=th.int)
        self.height, self.width = args.env_args['world_shape']

        # Observation properties
        self.watch_covered = getattr(args, "watch_covered", True)
        self.watch_surface = getattr(args, "watch_surface", True)
        self.observation_range = getattr(args, "observation_range", -1) # -1 = full observability

        # Reward function
        self.time_reward      = getattr(args, "reward_time", -0.1)
        self.collision_reward = getattr(args, "reward_collision", 0.0)
        self.new_cell_reward  = getattr(args, "reward_cell", 1.0)
        self.succes_reward    = getattr(args, "reward_succes", 0.0)
        self.invalid_reward   = getattr(args, "reward_invalid", 0.0)
        self.threat_reward    = getattr(args, "reward_threat", 1.0) # this constant is multiplied by the designed reward function


    """ When new perception is percepted, update the real state """
    def update_state(self, state, t_ep):
        if t_ep == 0:  # when the env time resets, a new episode has begun
            self.b_episode = (self.b_episode + 1) % self.buffer_size
            self.t = 0
        self.state = state.reshape(self.height, self.width, -1)
        
        # extract agents' locations from the state
        self.agents = th.stack(th.where(self.state[:, :, 0] > 0)).transpose(0, 1).cpu()
        identities = (self.state[self.agents[:, 0], self.agents[:, 1], 0].long() - 1).argsort()
        self.agents = self.agents[identities]

        return self._detect_interaction()
    
    """ Use the general state to create an observation for the agents """
    def get_obs_agent(self, agent_id):
        # Filter the grid layers that available to the agent
        watch = np.unique([0, self.watch_covered, 2 * self.watch_surface])

        # Observation-mode 1 - return the whole grid
        observation = self.state[:, :, watch].clone()
        agent_location = (self.agents[agent_id, 0], self.agents[agent_id, 1])

        # Remove agents' id from observation
        observation[:, :, 0] = (observation[:, :, 0] != 0)

        # Add agent's location (one-hot)
        one_hot = th.unsqueeze(th.zeros_like(observation[:, :, 0]), axis=2)
        one_hot[agent_location[0], agent_location[1]] = 1.0

        # First layer is one-hot of agent's location, the other are the state data, reshaped to 1D vector
        return th.dstack((one_hot, observation)).reshape(-1)

    # Use avail_actions to detect obstacles; this function avoid collisions (case that another agent has moved to adjacent cell)
    def get_avail_actions(self, agent_id, avail_actions):
        new_loc = self.agents[agent_id] + self.action_effect * avail_actions.transpose(0, 1).cpu()
        no_collision = (self.state[new_loc[:, 0], new_loc[:, 1], 0] != 0) * \
            (self.state[new_loc[:, 0], new_loc[:, 1], 0] != (agent_id + 1))
        return avail_actions * (~ no_collision)


    """ Simulate the result of action in the environment """
    # TODO - threats wasn't implemented (and some parts of the reward function)
    def _apply_action_on_state(self, agent_id, action, avail_actions, result=0):
        agent_location = self.agents[agent_id, :]
        if not avail_actions.flatten()[action]:
            action = 4 # Do nothing (does not support invalid/collision reward)

        # apply agent movement
        new_location = agent_location + self.action_effect[action]
        new_location %= th.tensor(self.state.shape[:2])

        if self.state[new_location[0], new_location[1], 0] == 0 and self.state[new_location[0], new_location[1], 2] == 0:
            self.state[agent_location[0], agent_location[1], 0] = 0.0
            self.state[new_location[0], new_location[1], 0] = agent_id + 1.0

        # mark new cell as covered
        new_cell = self.state[new_location[0], new_location[1], 1] == 0.0
        self.state[new_location[0], new_location[1], 1] = 1.0

        # check termination
        terminated = (th.sum(self.state[:, :, 1]) == self.height * self.width) or (self.t == self.episode_limit - 1)

        # calculate reward
        reward = self.time_reward + self.new_cell_reward * new_cell

        return reward, terminated