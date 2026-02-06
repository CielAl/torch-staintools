from dataclasses import dataclass

__all__ = ['PARAM']

@dataclass
class _Param:
    # L2 penalty for ridge initialization of codes Z
    INIT_RIDGE_L2: float = 1e-4

    # L2 weight decay term of dictionary W in the lasso loss of dictionary learning
    # DICT_WEIGHT_DECAY: float = 10e-10
    DICT_ITER_STEPS: int = 60

    # max concentration perc
    MAX_CONCENTRATION_PERC: int = 95
    # Optimization convergence tolerance
    OPTIM_DEFAULT_TOL: float = 1e-5
    # Lambda term in ISTA
    OPTIM_DEFAULT_SPARSE_LAMBDA: float = 1e-2
    # OPTIM_DEFAULT_SPARSE_CD_LAMBDA: float = 1
    # maxiter for sparse code
    OPTIM_SPARSE_DEFAULT_MAX_ITER: int = 50

    OPTIM_MOMEMTUM_RESTART_INTERVAL: int = 10

    # dict_magnitude = SOLVER_PINV_RTOL_SCALE * stain_mat.abs().amax(dim=(-1, -2))
    # atol = scale * quantile(dict_mag, q)
    SOLVER_PINV_ATOL_SCALE: float = 1e-2
    SOLVER_PINV_ATOL_QUANTILE: float = 0.25

PARAM: _Param = _Param()



