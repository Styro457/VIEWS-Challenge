import sys
from cachetools import LFUCache, cached

MAX_CACHE_SIZE = 50_000
FREQUENCY_THRESHOLD = 2

access_counts = {}

cache = LFUCache(maxsize=MAX_CACHE_SIZE, getsizeof=lambda v: sys.getsizeof(v))


def make_key(args, kwargs):
    # convert small lists to tuples, others remain the same
    key_args = tuple(tuple(a) if isinstance(a, list) else a for a in args)
    key_kwargs = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in kwargs.items())
    return (key_args, key_kwargs)

def frequency_cache(func, threshold=2):
    """
    Frequency-threshold caching annotation that saves results only
    after a specific number of calls
    """
    def wrapper(*args, **kwargs):
        key = make_key(args, kwargs)
        access_counts[key] = access_counts.get(key, 0) + 1

        if key in cache:
            return cache[key]

        result = func(*args, **kwargs)

        if access_counts[key] >= threshold:
            cache[key] = result

        return result
    return wrapper