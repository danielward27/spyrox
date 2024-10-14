from collections.abc import Callable

import equinox as eqx
from flowjax.distributions import AbstractDistribution
from jaxtyping import Array, PRNGKeyArray
from pyrox.program import AbstractProgram


class SimulatorToDistribution(AbstractDistribution):
    """Create a flowjax distribution from a simulator.

    We assume the simulator is JAX compatible by default. If you wish to use a non-JAX
    simulator, the simulator must be wrapped using ``jax.pure_callback``, ideally
    supporting broadcasting (see ``vmap_method`` argument to ``pure_callback``).

    Args:
        simulator: The simulator, accepting a key, returning a single simulation.
        shape: Output shape for simulator.
        cond_shape: Input shape for simulator.
    """

    simulator: Callable[[PRNGKeyArray, Array], Array]
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]

    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        raise NotImplementedError("Simulator does not have a tractable log prob.")

    def _sample(self, key: PRNGKeyArray, condition: Array):
        return self.simulator(key, condition)


# TODO this vs an unwrappable?
class AbstactProgramWithSurrogate(AbstractProgram):

    surrogate: eqx.AbstractVar[AbstractDistribution]
    simulator: eqx.AbstractVar[SimulatorToDistribution]
