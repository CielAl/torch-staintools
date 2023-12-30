from abc import ABC, abstractmethod
from typing import Callable, Optional, List, TypeVar, Generic, Hashable
from ..loggers import GlobalLoggers

logger = GlobalLoggers.instance().get_logger(__name__)
C = TypeVar('C')
V = TypeVar('V')


class Cache(ABC, Generic[C, V]):

    __size_limit: int
    data_cache: C

    @abstractmethod
    def __len__(self):
        ...

    @property
    def size_limit(self):
        return self.__size_limit

    def __init__(self, size_limit: int):
        self.__size_limit = size_limit
        self.data_cache = self._new_cache()

    @abstractmethod
    def query(self, key: Hashable):
        raise NotImplementedError

    @abstractmethod
    def is_cached(self, key: Hashable):
        raise NotImplementedError

    def __contains__(self, key: Hashable):
        return self.is_cached(key)

    @abstractmethod
    def write_to_cache(self, key: Hashable, value: V):
        raise NotImplementedError

    def get(self, key: Hashable, func: Optional[Callable], *func_args, **func_kwargs):
        if key in self:
            return self.query(key)
        # if not cached
        assert func is not None
        value = func(*func_args, **func_kwargs)
        if Cache.size_in_bound(len(self), 1, self.size_limit):
            self.write_to_cache(key, value)
        return value

    @abstractmethod
    def dump(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError

    def write_batch(self, keys: List[Hashable], batch: V):
        for k, b in zip(keys, batch):
            self.write_to_cache(k, b)

    @staticmethod
    def size_in_bound(current_size, in_data_size, limit_size):
        if limit_size < 0:
            return True

        return current_size + in_data_size <= limit_size

    def get_batch(self, keys: List[Hashable], func: Optional[Callable], *func_args, **func_kwargs) -> List[V]:
        """Batchified `get`

        The method assumes that the func callable would generate a whole batch of data each time.
        Might be useful if batchified processing is much faster than individually process all inputs.

        It is a hit only if all keys are cached.

        Args:
            keys:
            func
            *func_args:
            **func_kwargs:

        Returns:

        """
        hit = all([k in self for k in keys])
        if hit:
            logger.debug('hit')
            return [self.get(k, func=None) for k in keys]
        assert func is not None
        logger.debug('miss - evaluate function to get value')
        batch = func(*func_args, **func_kwargs)
        assert len(keys) == len(batch)
        if Cache.size_in_bound(len(self), len(keys), self.size_limit):
            logger.debug(f'{len(self)} - add new cache to {keys[0:3]}...')
            self.write_batch(keys, batch)
        return batch

    @abstractmethod
    def _new_cache(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs):
        ...
