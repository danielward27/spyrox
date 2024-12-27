from functools import partial

import equinox as eqx
import jax
import jax.random as jr
import numpyro
from flowjax.distributions import AbstractDistribution
from flowjax.experimental.numpyro import _RealNdim


class VmapDistribution(numpyro.distributions.Distribution):
    """Convert a flowjax distribution to a numpyro distribution with a batch dimension.

    For now this assumes all parameters are vectorized, but this is likely to change
    in the future. It also does not handle Transformed distributions well,
    (e.g. in terms of efficiency of sample_and_log_prob).

    Args:
        dist: AbstractDistribution, created e.g. with eqx.filter_vmap, to have a
            batch dimension.
        batch_size: The size of the leading dimension.
    """

    dist: AbstractDistribution
    batch_size: int

    def __init__(
        self,
        dist: AbstractDistribution,
        batch_size: int,
    ):
        self.dist = dist
        self.batch_size = batch_size
        self.support = _RealNdim(dist.ndim)
        super().__init__((batch_size,), dist.shape)

    def sample(self, key, sample_shape=()):
        # TODO remove when old-style keys fully deprecated
        if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
            key = jr.wrap_key_data(key)

        # Out axes ensures (*sample, *batch, *event), not (*batch, *sample, *event)
        out_axis = -self.dist.ndim - 1

        @partial(eqx.filter_vmap, out_axes=out_axis)
        def _sample_dist(key, d):
            return d.sample(key, sample_shape)

        keys = jr.split(key, self.batch_size)
        return _sample_dist(keys, self.dist)

    def log_prob(self, value):
        vmap_dim = -self.dist.ndim - 1

        @partial(eqx.filter_vmap, in_axes=(eqx.if_array(0), eqx.if_array(vmap_dim)))
        def _log_prob(d, value):
            return d.log_prob(value)

        return _log_prob(self.dist, value)

    # TODO no sample and log_prob / with intermediates?
