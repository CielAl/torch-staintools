from dataclasses import dataclass

__all__ = ['CONST']

@dataclass
class _Config:
    # L2 penalty for ridge initialization of codes Z
    INIT_RIDGE_L2: float = 1e-4
    # L2 weight decay term of dictionary W in the lasso loss of dictionary learning
    DICT_WEIGHT_DECAY: float = 10e-10
    # Whether the code is persisted in the iterative procedure of dictionary learning
    DICT_PERSIST_CODE: bool = True
    # Whether to Enforce Positive Dictionary / Stain Matrix
    DICT_POSITIVE_DICTIONARY: bool = True
    # Whether to Enforce Positive Code / Concentration
    DICT_POSITIVE_CODE: bool = True


CONST = _Config()



