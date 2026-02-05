from dataclasses import dataclass

__all__ = ['CONFIG']

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

    ENABLE_VECTORIZE: bool = True

CONFIG: _Config = _Config()



