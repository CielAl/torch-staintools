from ..cache.tensor_cache import TensorCache
import torch
from typing import Optional, List, Hashable, Callable
from abc import abstractmethod
from ..functional.utility.implementation import default_device, default_rng
from ..loggers import GlobalLoggers

logger = GlobalLoggers.instance().get_logger(__name__)
# @staticmethod
# def new_cache(shape):
#     """
#     Args:
#         shape:
#
#     Returns:
#
#     """
#     # Todo map the key to the corresponding cached data -- cached in file or to memory?
#     #
#     shared_array_base = mp.Array(ctypes.c_float, reduce(mul, shape))
#     shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
#     shared_array = shared_array.reshape(*shape)
#     return shared_array

#


class CachedRNGModule(torch.nn.Module):
    """Optionally cache the stain matrices and manage the rng

    Note that using nn.Module.to(device) to move the module across GPU/cpu device will reset the states.


    """
    device: torch.device
    _tensor_cache: TensorCache
    CACHE_FIELD: str = '_tensor_cache'
    rng: Optional[torch.Generator]

    def _tensor_cache_helper(self) -> Optional[TensorCache]:
        return getattr(self, CachedRNGModule.CACHE_FIELD)

    def cache_initialized(self):
        return hasattr(self, CachedRNGModule.CACHE_FIELD) and self._tensor_cache_helper() is not None

    @property
    def tensor_cache(self) -> Optional[TensorCache]:
        return self._tensor_cache_helper()

    @staticmethod
    def _init_cache(use_cache: bool, cache_size_limit: int, device: Optional[torch.device] = None,
                    load_path: Optional[str] = None) -> Optional[TensorCache]:
        if not use_cache:
            return None
        return TensorCache.build(size_limit=cache_size_limit, device=device, path=load_path)

    @staticmethod
    def _rng_to(rng: Optional[torch.Generator], device: torch.device):
        if rng is None:
            return None
        new_rng = torch.Generator(device)
        old_state = rng.get_state()  # byte tensor
        if new_rng.get_state().size() == old_state.size():
            new_rng.set_state(old_state)
        else:
            # can only copy the initial seed
            new_rng.manual_seed(rng.initial_seed())
        return new_rng

    def to(self, device: torch.device):
        self.device = device
        self.rng = CachedRNGModule._rng_to(self.rng, device)
        if self.cache_initialized():
            self.tensor_cache.to(device)
        return super().to(device)

    @property
    def cache_size_limit(self) -> int:
        if self.cache_initialized():
            return self.tensor_cache.size_limit
        return 0

    def dump_cache(self, path: str):
        assert self.cache_initialized()
        self.tensor_cache.dump(path)

    # @staticmethod
    # def _stain_mat_kwargs_helper(luminosity_threshold,
    #                              num_stains,
    #                              regularizer,
    #                              **stain_mat_kwargs):
    #     arg_dict = {
    #         'luminosity_threshold': luminosity_threshold,
    #         'num_stains': num_stains,
    #         'regularizer': regularizer,
    #     }
    #     stain_mat_kwargs = {k: v for k, v in stain_mat_kwargs.items()}
    #     stain_mat_kwargs.update(arg_dict)
    #     return stain_mat_kwargs

    @staticmethod
    def tensor_from_cache_helper(cache: TensorCache, *,
                                 cache_keys: List[Hashable],
                                 func_partial: Callable,
                                 target) -> torch.Tensor:

        stain_mat_list = cache.get_batch(cache_keys, func_partial, target)
        if isinstance(stain_mat_list, torch.Tensor):
            return stain_mat_list

        return torch.stack(stain_mat_list, dim=0)

    def tensor_from_cache(self,
                          *,
                          cache_keys: Optional[List[Hashable]],
                          func_partial: Callable,
                          target) -> torch.Tensor:
        if cache_keys is not None and not self.cache_initialized():
            logger.warning(f"Cache keys are given but the cache is not initialized: {cache_keys[:3]} etc..")

        if not self.cache_initialized() or cache_keys is None:
            logger.debug(f'{self.cache_initialized()} + {cache_keys is None} - no cache')
            return func_partial(target)
        # if using cache
        assert self.cache_initialized(), f"Attempt to fetch data from cache but cache is not initialized"
        assert cache_keys is not None, f"Attempt to fetch data from cache but key is not given"
        # move fetched stain matrix to the same device of the target
        logger.debug(f"{cache_keys[0:3]}. cache initialized")
        return CachedRNGModule.tensor_from_cache_helper(cache=self.tensor_cache, cache_keys=cache_keys,
                                                        func_partial=func_partial,
                                                        target=target).to(target.device)

    def __init__(self, cache: Optional[TensorCache], device: Optional[torch.device],
                 rng: Optional[int | torch.Generator]):
        super().__init__()

        self._tensor_cache = cache
        self.device = default_device(device)
        self.rng = default_rng(rng, self.device)

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs):
        ...
