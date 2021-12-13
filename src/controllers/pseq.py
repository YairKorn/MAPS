from .basic_controller import BasicMAC
from src.action_model import REGISTRY as model_REGISTRY

class PSeqMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

        self.action_model = model_REGISTRY[args.environment](args)

    def forward(self, ep_batch, t, test_mode=False):
        
        return super().forward(ep_batch, t, test_mode=test_mode)