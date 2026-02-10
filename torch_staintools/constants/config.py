from dataclasses import dataclass

__all__ = ['CONFIG']

from typing import Optional


@dataclass
class _Config:
    # Whether the code is persisted in the iterative procedure of dictionary learning
    DICT_PERSIST_CODE: bool = True
    # Whether to Enforce Positive Dictionary / Stain Matrix
    DICT_POSITIVE_DICTIONARY: bool = True
    # Whether to Enforce Positive Code / Concentration
    DICT_POSITIVE_CODE: bool = True

    # force the augmentation results in positive C only
    AUG_POSITIVE_CONCENTRATION: bool = True
    # Whether to enable torch.compile (currently only the dictionary learning is affected)
    ENABLE_COMPILE: bool = True

    # basically torch.compile(dynamic=ENABLE_DYNAMIC_SHAPE)
    # Must be gauged before invoking any functions to be compiled
    ENABLE_DYNAMIC_SHAPE: Optional[bool] = True

    # whether to vectorize the batchified procedure
    # can be gauged anytime.
    ENABLE_VECTORIZE: bool = True

CONFIG: _Config = _Config()



