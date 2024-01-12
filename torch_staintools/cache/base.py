from abc import ABC, abstractmethod
from typing import Callable, Optional, List, TypeVar, Generic, Hashable
import os
from ..loggers import GlobalLoggers

logger = GlobalLoggers.instance().get_logger(__name__)
C = TypeVar('C')
V = TypeVar('V')


class Cache(ABC, Generic[C, V]):
    """A simple abstraction of cache.

    """

    __size_limit: int
    data_cache: C

    @abstractmethod
    def __len__(self):
        """current size of cache

        Returns:

        """
        raise NotImplementedError

    @property
    def size_limit(self):
        return self.__size_limit

    def __init__(self, size_limit: int):
        """

        Args:
            size_limit: limit of cache size by number of entries (no greater than). If negative then no size limit is
                enforced.
        """
        self.__size_limit = size_limit
        self.data_cache = self._new_cache()

    @abstractmethod
    def query(self, key: Hashable):
        """Behavior of how to read data under key in cache. Used in `get` and `get_batch`

        Args:
            key:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def is_cached(self, key: Hashable):
        """whether the key already stores a value.

        Args:
            key:  key to query

        Returns:
            bool of whether the corresponding key already stores a value.
        """

        raise NotImplementedError

    def __contains__(self, key: Hashable):
        """For simplicity. same as `is_cached`

        Args:
            key: key to query

        Returns:
            bool of whether the corresponding key already stores a value.
        """
        return self.is_cached(key)

    @abstractmethod
    def _write_to_cache_helper(self, key: Hashable, value: V):
        """Write the data (value) to the given address (key) in the cache

        Args:
            key: any hashable that points the data to the address in the cache
            value: value of the data to cache

        Returns:

        """
        raise NotImplementedError

    def write_to_cache(self, key: Hashable, value: V):
        """Write the data (value) to the given address (key) in the cache

        Args:
            key: any hashable that points the data to the address in the cache
            value: value of the data to cache

        Returns:

        """
        if not Cache.size_in_bound(len(self), 1, self.size_limit):
            return
        # logger.debug(f"key to write - parent: {key}")
        self._write_to_cache_helper(key, value)

    def get(self, key: Hashable, func: Optional[Callable], *func_args, **func_kwargs):
        """Get the data cached under key.

        If the corresponding data of key is not yet cached, it will be computed by the
        func(`*func_args`, `**func_kwargs`) and the results will be cached if the remaining size is sufficient.

        Args:
            key: the address of the data in cache
            func: callable to evaluate the new data to cache if not yet cached under `key`
            *func_args: positional arguments of func
            **func_kwargs: keyword arguments of func

        Returns:

        """
        if key in self:
            return self.query(key)
        # if not cached
        assert func is not None
        value = func(*func_args, **func_kwargs)
        # if Cache.size_in_bound(len(self), 1, self.size_limit):
        self.write_to_cache(key, value)
        return value

    def dump(self, path: str, force_overwrite: bool = False):
        """ Dump the cached data to the local file system.

        Args:
            path: output filename
            force_overwrite: whether to force overwriting the existing file on path
        Returns:

        """
        if os.path.exists(path) and not force_overwrite:
            return
        self._dump_helper(path)

    @abstractmethod
    def _dump_helper(self, path: str):
        """ To implement: dump the cached data to the local file system.

        Args:
            path: output filename

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        """Load cache from the local file system.

        Args:
            path:

        Returns:

        """
        raise NotImplementedError

    def write_batch(self, keys: List[Hashable], batch: V):
        """Write a batch of data to the cache.

        Args:
            keys: list of keys corresponding to individual data points in the batch.
            batch: batch data to cache.

        Returns:

        """
        # if not Cache.size_in_bound(len(self), len(keys), self.size_limit):
        #     return
        logger.debug(f'{len(self)} - add new cache to {keys[0:3]}...')
        for k, b in zip(keys, batch):
            self.write_to_cache(k, b)

    @staticmethod
    def size_in_bound(current_size, in_data_size, limit_size):
        """Check whether the size is still in-bound with new data added into cache

        Args:
            current_size: current size of cache
            in_data_size: size of new data
            limit_size: current size limit (no greater than). If negative then no size limit is enforced.

        Returns:
            bool. If the size is still in-bound with new data loaded into the cache.
        """
        if limit_size < 0:
            return True
        # logger.debug(f"check: {current_size} + {in_data_size} <= {limit_size}")
        return current_size + in_data_size <= limit_size

    def get_batch(self, keys: List[Hashable], func: Optional[Callable], *func_args, **func_kwargs) -> List[V]:
        """Batchified `get`

        The method assumes that the func callable would generate a whole batch of data each time.
        Might be useful if batchified processing is much faster than individually process all inputs
        (e.g., cuda tensors processed by nn.Module)

        It is a hit only if all keys are cached.

        Args:
            keys: list of keys corresponding to the batch input.
            func: function to generate the data if the corresponding entry is not cached.
            *func_args: positional args for the func.
            **func_kwargs: keyword args for the func.

        Returns:
            List of queried results.
        """
        hit = all([k in self for k in keys])
        if hit:
            logger.debug('hit')
            return [self.get(k, func=None) for k in keys]
        assert func is not None
        logger.debug('miss - evaluate function to get value')
        batch = func(*func_args, **func_kwargs)
        assert len(keys) == len(batch)

        self.write_batch(keys, batch)
        return batch

    @abstractmethod
    def _new_cache(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs):
        ...
