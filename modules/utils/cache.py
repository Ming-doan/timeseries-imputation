"""
Cache module for saving and loading data.
"""
import sys
from typing import Callable, Any


class Cache:
    """
    Cache class for saving and loading data.

    If you want to use the cache, you must implement the `on_load` and `on_save` function.

    If live_cache, cache will be saved in the memory.
    """

    def __init__(self, on_load: Callable = None, on_save: Callable = None, live_cache: bool = False):
        self.on_load = on_load
        self.on_save = on_save
        self.live_cache = live_cache

        if not self.live_cache:
            if self.on_load is None:
                raise ValueError(
                    'If `live_cache` is True, `on_load` must be implemented.')
            if self.on_save is None:
                raise ValueError(
                    'If `live_cache` is True, `on_save` must be implemented.')

        self.memory = {}

    def get(self, cache_id: str) -> Any:
        """
        Get the data from the cache.
        """
        if self.live_cache:
            return self.memory[cache_id]
        return self.on_load(cache_id)

    def set(self, cache_id: str, data: Any):
        """
        Set the data to the cache.
        """
        if self.live_cache:
            self.memory[cache_id] = data
        else:
            self.on_save(cache_id, data)

    def auto(self, cache_id: str, data: Any) -> tuple[bool, Any]:
        """
        Load the data from the cache.
        If cache hit, return the cached data.
        If cache miss, return the data and save it to the cache.
        """
        try:
            return True, self.get(cache_id)
        except KeyError as _:
            self.set(cache_id, data)
            return False, data

    def memory_usage(self) -> int:
        """
        Get the memory usage.
        """
        return sum(sys.getsizeof(v) for v in self.memory.values())

    def flush(self):
        """
        Flush the cache.
        """
        self.memory = {}
