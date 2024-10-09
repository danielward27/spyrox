from collections.abc import Callable

from flowjax.distributions import AbstractDistribution
from jaxtyping import Array, PRNGKeyArray


class SimulatorDistribution(AbstractDistribution):
    """Create a distribution from a simulator.

    Args:
        simulator: The simulator, accepting a key, returning a simulation.
        shape: Output shape for simulator.
        cond_shape: Input shape for simulator.
        surrogate: Surrogate distribution, to use to replace log_prob method.
            Defaults to None (raising a not implemented error).
    """

    simulator: Callable[[PRNGKeyArray, Array], Array]
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    surrogate: AbstractDistribution | None = None

    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        if self.surrogate is None:
            raise NotImplementedError(
                "Simulator does not have log prob. Consider using a surrogate.",
            )
        return self.surrogate._log_prob(x, condition)

    def _sample(self, key: PRNGKeyArray, condition: Array):
        return self.simulator(key, condition)
