import numpy as np
from .basic_model import BasicAM

class StagHunt(BasicAM):
    def __init__(self, args):
        super().__init__(args)

        # env specific properties
        self.toroidal = args.env_args['toroidal']
        self.actions = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int16)

        # world initializing
        self.x_max, self.y_max = args.env_args['world_shape']


    def update_state(self, input_state):
        self.state, self.state_prob = input_state[0].reshape((self.x_max, self.y_max, 3)), np.ones(1)
        return self._detect_interaction(self.state)

    def step(self, actor, actions):
        # if action is a list, more than one agent execute an action simultaneously
        if type(actor) is list:
            pass

        # otherwise, it's one agent action
        else:
            pass
    
    def get_obs_agent(self, agent_id):
        pass