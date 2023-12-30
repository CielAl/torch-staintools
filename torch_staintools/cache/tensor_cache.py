from .base import Cache
from typing import Dict, Hashable, Optional
import torch
from ..functional.utility.implementation import default_device


class TensorCache(Cache[Dict[Hashable, torch.Tensor], torch.Tensor]):
    # perhaps tensordict can be used in future
    data_cache: Dict[Hashable, torch.Tensor]
    __size_limit: int
    device: torch.device

    def __len__(self):
        return len(self.data_cache)

    def query(self, key):
        return self.data_cache[key]

    def is_cached(self, key):
        return key in self.data_cache

    @staticmethod
    def validate_value_type(value: torch.Tensor):
        assert isinstance(value, torch.Tensor)
        return value

    def write_to_cache(self, key, value: torch.Tensor):
        value = TensorCache.validate_value_type(value)
        self.data_cache[key] = value.to(self.device)

    @staticmethod
    def _to_device(data_cache: Dict[Hashable, torch.Tensor], device: torch.device, dict_inplace: bool = True):
        if not dict_inplace:
            return {k: v.detach().to(device) for k, v in data_cache.items()}
        for k in data_cache:
            data_cache[k] = data_cache[k].to(device)
        return data_cache

    def dump(self, path: str):
        cpu_cache = TensorCache._to_device(self.data_cache, torch.device('cpu'), dict_inplace=False)
        torch.save(cpu_cache, path)

    def load(self, path: str):
        # load
        data_dict = torch.load(path)
        data_dict = TensorCache._to_device(data_dict, self.device, dict_inplace=True)
        self.data_cache.update(data_dict)
        self.__size_limit = len(self.data_cache)

    def _new_cache(self):
        return dict()

    def __init__(self, size_limit: int, device: Optional[torch.device] = None):
        super().__init__(size_limit)
        self.device = default_device(device)

    @classmethod
    def build(cls, *, size_limit: int = -1, device: Optional[torch.device] = None, path: Optional[str] = None):
        new_cache = cls(size_limit=size_limit, device=device)
        if path is not None:
            new_cache.load(path)
        return new_cache

    def to(self, device: torch.device):
        if self.device == device:
            return
        self.data_cache = TensorCache._to_device(self.data_cache, device, dict_inplace=True)
        self.device = device
        return self
