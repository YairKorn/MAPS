import numpy as np  
from .model import ActionModel

class StagHunt(ActionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.no_agents = self.args.n_agents
        self.toroidal = self.args.toroidal

        # env specific properties
        self.actions = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int16)

        # world initializing
        self.x_max, self.y_max = self.args.world_shape


    def update_state(self, state):
        return super().update_state(state)

    def step(self, action):
        # if action is a list, more than one agent execute an action simultaneously
        if type(action) is list:
            pass

        # otherwise, it's one agent action
        else:
            pass
    
    def get_obs_agent(self, agent_id):
        pass