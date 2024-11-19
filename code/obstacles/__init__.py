# Public API
from .abstract_road import AbstractRoad
from .road_collection import roads

__all__ = ['AbstractRoad', 'roads']

# Prevent access to _internal modules
import sys

for name in list(sys.modules):
    if name.startswith(f"{__name__}._internal"):
        del sys.modules[name]