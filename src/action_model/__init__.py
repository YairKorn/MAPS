REGISTRY = {}

from .stag_hunt import StagHunt
REGISTRY["stag_hunt"] = StagHunt

from .adv_coverage import AdvCoverage
REGISTRY["mrac"] = AdvCoverage

from .hunt_trip import HuntingTrip
REGISTRY["hunt_trip"] = HuntingTrip

from .gold_coverage import GoldCoverage
REGISTRY["gold"] = GoldCoverage
