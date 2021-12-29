from .model_v1 import ActionModel

class AdvCoverage(ActionModel):
    
    
    def _apply_action_on_state(self, state, action, result=0):
        return state