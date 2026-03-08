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
# period(8) + group(19) + degree(8) + hybridisation(7) + charge(6) + hs(6) + aromatic(1) + in_ring(1)
ATOM_FEAT_DIM: int = 56
# bond_type(5) + conjugated(1) + in_ring(1) + rotatable(1) + stereo(4)
BOND_FEAT_DIM: int = 12
