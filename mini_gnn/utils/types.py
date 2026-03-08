from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

# ---- basic array aliases ----
Array    = jax.Array          # JAX device array
NpArray  = np.ndarray         # NumPy host array
PyTree   = Any                # generic JAX pytree (params, opt_state, …)

# ---- dimension constants (fixed widths used everywhere) ----
# These match featurizers.py. Centralised here so JIT never sees shape mismatches.
ATOM_FEAT_DIM: int = 72
BOND_FEAT_DIM: int = 16
