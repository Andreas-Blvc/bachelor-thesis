# Public API
from .interface import AbstractRoad
from .road_collection import roads

__all__ = ['AbstractRoad', 'roads']


import warnings
warnings.warn("the obstacle module is deprecated", DeprecationWarning,
              stacklevel=2)

# Prevent access to _internal modules
import sys

for name in list(sys.modules):
    if name.startswith(f"{__name__}._internal"):
        del sys.modules[name]