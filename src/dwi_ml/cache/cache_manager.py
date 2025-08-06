# -*- coding: utf-8 -*-

from collections import deque


class CacheManager(object):
    """Basic CacheManager interface"""

    def __init__(self, cache_size: int):
        self._cache_size = cache_size
        self._cache = None
        self._queue = None

    def __contains__(self, item):
        return item in self._cache

    def __getitem__(self, item):
        return self._cache[item]

    def __setitem__(self, key, value):
        raise NotImplementedError

    def empty_cache(self):
        raise NotImplementedError


class SingleThreadCacheManager(CacheManager):
    """A single-thread FIFO dictionary cache"""

    def __init__(self, cache_size: int):
        super(SingleThreadCacheManager, self).__init__(cache_size)
        self._cache = dict()
        self._queue = deque()

    def __setitem__(self, key, value):
        if len(self._queue) >= self._cache_size:
            to_delete = self._queue.popleft()
            del self._cache[to_delete]
        self._queue.append(key)
        self._cache[key] = value

    def empty_cache(self):
        self._cache = dict()
        self._queue = deque()
