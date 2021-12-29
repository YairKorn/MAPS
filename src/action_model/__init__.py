REGISTRY = {}

from .stag_hunt import StagHunt
REGISTRY["stag_hunt"] = StagHunt

from .adv_coverage import AdvCoverage
REGISTRY["mrac"] = AdvCoverage

from .adv_coverage import AdvCoverage
REGISTRY["mrac_e1"] = AdvCoverage