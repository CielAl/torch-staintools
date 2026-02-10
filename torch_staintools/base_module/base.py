import warnings

from ..cache.tensor_cache import TensorCache
import torch
from typing import Optional, List, Hashable, Callable
from abc import abstractmethod

from ..functional.stain_extraction.extractor import StainExtraction
from ..functional.utility.implementation import default_device, default_rng
from ..hash.key import key_from_od
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
    _rng: Optional[torch.Generator]

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

    def load_cache(self, path: str):
        if not self.cache_initialized():
            self._tensor_cache = self._init_cache(use_cache=True, cache_size_limit=-1, device=self.device)
        self._tensor_cache.load(path)

#############################3


    def _get_batch_hit_all(self, keys: List[Hashable]) -> torch.Tensor:
        logger.debug('hit')
        return self.tensor_cache.get_batch_hit(keys)

    def _get_batch_miss_all(self, keys: List[Hashable], get_stain_mat: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert get_stain_mat is not None
        logger.debug('miss - evaluate function to get value')
        # batch = get_stain_mat(target, mask)
        # assert len(keys) == len(batch)
        return self.tensor_cache.get_batch_miss(keys, get_stain_mat, target, mask)

    def _get_batch_mixed(self, keys: List[Hashable],
                         hit_indices: List[int],
                         miss_indices: List[int],
                         get_stain_mat: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                         target, mask) -> torch.Tensor:
        logger.debug(f'partial miss - {len(miss_indices)} out of {len(keys)}')
        assert len(hit_indices) > 0
        assert len(miss_indices) >0
        assert len(hit_indices) + len(miss_indices) == len(keys)
        assert target.shape[0] == len(keys) == mask.shape[0]
        idx_cached = torch.tensor(hit_indices, device=target.device, dtype=torch.long)
        idx_new = torch.tensor(miss_indices, device=target.device, dtype=torch.long)

        # missed branch
        target_new = target.index_select(0, idx_new)
        mask_new = mask.index_select(0, idx_new)
        key_missed = [keys[i] for i in miss_indices]
        sm_new = self._get_batch_miss_all(key_missed, get_stain_mat, target_new, mask_new)
        self.tensor_cache.write_batch(key_missed, sm_new)

        # hit branch
        key_hit = [keys[i] for i in hit_indices]
        sm_hit = self._get_batch_hit_all(key_hit)
        #
        output_shape = (target.shape[0], *sm_new.shape[1:])
        sm_out = torch.empty(output_shape, device=target.device, dtype=target.dtype)
        sm_out.index_copy_(0, idx_new, sm_new)
        sm_out.index_copy_(0, idx_cached, sm_hit)
        return sm_out.contiguous()

    def get_batch(self, keys: List[Hashable],
                  get_stain_mat: StainExtraction,
                  target: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor] | torch.Tensor:
        """Batchified `get`

        The method assumes that the func callable would generate a whole batch of data each time.
        Might be useful if batchified processing is much faster than individually process all inputs
        (e.g., cuda tensors processed by nn.Module)

        It is a hit only if all keys are cached.

        Args:
            keys: list of keys corresponding to the batch input.
            get_stain_mat: function to generate the data if the corresponding entry is not cached.
            target: target tensor. Potentially in OD space.
            mask: mask the background pixel to 0. foreground regions are 1.

        Returns:
            List of queried results.
        """
        hit_indices = [idx for idx, k in enumerate(keys) if k in self.tensor_cache]
        miss_indices = [idx for idx, k in enumerate(keys) if not k in self.tensor_cache]
        hit_all = len(hit_indices) == len(keys)
        miss_all = len(hit_indices) == 0
        # if all match then fetch everything
        if hit_all:
            return self._get_batch_hit_all(keys)
        # if all miss
        if miss_all:
            return self._get_batch_miss_all(keys, get_stain_mat, target, mask)
        # partial hit
        assert len(miss_indices) > 0
        return self._get_batch_mixed(keys, hit_indices, miss_indices,
                                     get_stain_mat=get_stain_mat, target=target, mask=mask)



    def stain_mat_cached_helper(self,
                                cache_keys: List[Hashable],
                                get_stain_mat:  StainExtraction,
                                target: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:

        stain_mats = self.get_batch(cache_keys, get_stain_mat, target, mask)
        if isinstance(stain_mats, torch.Tensor):
            # already a tensor, possibly all_miss.
            return stain_mats
        return CachedRNGModule.collect(stain_mats)


    def default_hash(self, cache_keys: Optional[List[Hashable]],
                     target: torch.Tensor, mask: torch.Tensor) -> Optional[List[Hashable]]:
        if cache_keys is not None:
            assert len(cache_keys) == target.shape[0]
            return cache_keys
        if not self.cache_initialized():
            return None
        return key_from_od(target, mask)
    ##########

    def stain_mat_cached(self,
                         *,
                         cache_keys: Optional[List[Hashable]],
                         get_stain_mat: StainExtraction,
                         target: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        cache_keys = self.default_hash(cache_keys, target, mask)
        if cache_keys is not None and not self.cache_initialized():
            logger.warning(f"Cache keys are given but the cache is not initialized: {cache_keys[:3]} etc..")

        # both are mandatory. If either is not ready then bypass caching.
        if not self.cache_initialized() or cache_keys is None:
            logger.debug(f'{self.cache_initialized()} + {cache_keys is None} - no cache')
            return get_stain_mat(target, mask)
        # if using cache
        assert self.cache_initialized(), f"Attempt to fetch data from cache but cache is not initialized"
        assert cache_keys is not None, f"Attempt to fetch data from cache but key is not given"
        # move fetched stain matrix to the same device of the target
        logger.debug(f"{cache_keys[0:3]}. cache initialized")
        return self.stain_mat_cached_helper(cache_keys=cache_keys,
                                            get_stain_mat=get_stain_mat,
                                            target=target, mask=mask).to(target.device)

    def __init__(self, cache: Optional[TensorCache], device: Optional[torch.device],
                 rng: Optional[int | torch.Generator]):
        super().__init__()

        self._tensor_cache = cache
        self.device = default_device(device)
        self._rng = default_rng(rng, self.device)
        if self._rng is not None:
            warnings.warn(f"A custom RNG is passed and may cause graph break if torch.compile is used."
                          f"Consider fixing random states globally instead.")

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, rng: torch.Generator):
        self._rng = default_rng(rng, self.device)

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs):
        ...
