from .base import Cache
from typing import Dict, Hashable, Optional
import torch
import numpy as np
from ..functional.utility.implementation import default_device
from ..loggers import GlobalLoggers
logger = GlobalLoggers.instance().get_logger(__name__)


class TensorCache(Cache[Dict[Hashable, torch.Tensor], torch.Tensor]):
    """An implementation of Cache specifically for tensor using a built-in dict.

    For now, it is used to store stain matrices directly on CPU or GPU memory since stain matrices are typically small
    (e.g., 2x3 for mapping between H&E and RGB).

    Size of concentrations, however, are proportionally to number of pixels x num_stains, therefore it might be better
    to be cached on the local file system.

    """
    # perhaps tensordict can be used in future
    data_cache: Dict[Hashable, torch.Tensor]
    __size_limit: int
    device: torch.device

    def __len__(self):
        """Size of cache given by number of entries stored.

        Returns:
            int:

        """
        return len(self.data_cache)

    def query(self, key) -> torch.Tensor:
        """Implementation of abstract method: query

        Read from dict directly

        Args:
            key:

        Returns:
            queried output

        Raises:
            KeyError.
        """
        return self.data_cache[key]

    def is_cached(self, key):
        return key in self.data_cache

    @staticmethod
    def validate_value_type(value: torch.Tensor | np.ndarray):
        """Helper function to validate the input.

         Must be a torch.Tensor. If it is a numpy ndarray, it will be converted to tensor.

        Args:
            value: value to validate

        Returns:
            torch.Tensor.

        Raises:
            AssertionError if the output is not a torch.Tensor
        """
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)

        assert isinstance(value, torch.Tensor), f"Expect tensor, got: {type(value)}"
        return value

    def _write_to_cache_helper(self, key, value: torch.Tensor):
        """Write the value into the key in cache. Will be moved to the specified device (GPU/CPU) during the procedure.

        Args:
            key: key to write
            value: value to write

        Returns:

        """
        value = TensorCache.validate_value_type(value)
        # logger.debug(f"key={key} - write")
        self.data_cache[key] = value.to(self.device)

    @staticmethod
    def _to_device(data_cache: Dict[Hashable, torch.Tensor], device: torch.device, dict_inplace: bool = True):
        """Helper function to move all cached tensors to the specified device

        Args:
            data_cache: the dict to operate on.
            device: target device. Note if a tensor is already on the target device, tensor.to(device) will be a no-op.
            dict_inplace: whether to move the tensors inplace of the same dict, or create a new dict to store moved
                tensors.

        Returns:
            the original (dict_inplace=True) or the new dict (dict_inplace=False) to store the moved tensors.
        """
        if not dict_inplace:
            return {k: v.detach().to(device) for k, v in data_cache.items()}
        for k in data_cache:
            data_cache[k] = data_cache[k].to(device)
        return data_cache

    def _dump_helper(self, path: str):
        """Dump the dict to the local file system.

        Note: A copy of the dict will be created, with all stored tensors copied to CPU. Dumped tensors are all
        CPU tensors.

        Args:
            path: file path to dump.
        Returns:

        """
        cpu_cache = TensorCache._to_device(self.data_cache, torch.device('cpu'), dict_inplace=False)
        torch.save(cpu_cache, path)

    def load(self, path: str):
        """Load cache from the local file system.

        Keys will be updated. Cached data already in memory will be overwritten if the same key existing in the
        dumped cache file to load from. Cached data that do not exist in the dumped cache file (by key) will not be
        affected.

        Args:
            path: file path to the local cache file to load.

        Returns:

        """
        # load
        data_dict = torch.load(path)
        data_dict = TensorCache._to_device(data_dict, self.device, dict_inplace=True)
        self.data_cache.update(data_dict)
        self.__size_limit = len(self.data_cache)

    def _new_cache(self) -> Dict:
        """Implementation of creating new cache - built-in dict.

        Returns:
            A new empty dict.
        """
        return dict()

    def __init__(self, size_limit: int, device: Optional[torch.device] = None):
        super().__init__(size_limit)
        self.device = default_device(device)

    @classmethod
    def build(cls, *, size_limit: int = -1, device: Optional[torch.device] = None, path: Optional[str] = None):
        """Factory builder.

        Args:
            size_limit: limit of the cache size by number of entries (no greater than number of keys). Negative value
                means no limit will be enforced.
            device: which device (CPU or GPUs) to store the tensor. If None then by default it will be set as
                torch.device('cpu').
            path: If specified, previously dumped cache file will be loaded from the path.

        Returns:

        """
        new_cache = cls(size_limit=size_limit, device=device)
        if path is not None:
            new_cache.load(path)
        return new_cache

    def to(self, device: torch.device):
        """Move the cache to the specified device. Simulate torch.nn.Module.to and torch.Tensor.to.

        The dict itself will be reused but the corresponding tensors stored in the dict might be copied to the target
        device if they are not already on the target device.

        Args:
            device: Target device

        Returns:
            self.
        """
        if self.device == device:
            return
        self.data_cache = TensorCache._to_device(self.data_cache, device, dict_inplace=True)
        self.device = device
        return self
