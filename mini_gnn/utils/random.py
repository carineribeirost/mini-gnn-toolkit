import jax
from mini_gnn.utils.types import Array


def make_key(seed: int) -> Array:
    return jax.random.PRNGKey(seed)


def split(key: Array) -> tuple[Array, Array]:
    """Return (new_key, subkey). Always use new_key for the next call."""
    return jax.random.split(key)


def split_n(key: Array, n: int) -> tuple[Array, list[Array]]:
    """Return (new_key, [subkey_0, ..., subkey_{n-1}])."""
    keys = jax.random.split(key, n + 1)
    return keys[0], list(keys[1:])
