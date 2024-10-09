"""A collection of tasks for validating model performance."""

# %%
from functools import partial

import equinox as eqx
import flowjax.distributions as dist
import jax.numpy as jnp
import jax.random as jr
from diffrax import (
    ControlTerm,
    Euler,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)
from flowjax import bijections as bij
from flowjax.experimental.numpyro import sample
from flowjax.flows import block_neural_autoregressive_flow, masked_autoregressive_flow
from jax import Array
from jaxtyping import PRNGKeyArray, ScalarLike
from numpyro import deterministic
from pyrox.program import AbstractProgram

from spyrox.simulator import SimulatorDistribution


class SIRSDEModel(AbstractProgram):
    simulator: SimulatorDistribution

    def __init__(
        self,
        key: PRNGKeyArray,
    ):
        surrogate = block_neural_autoregressive_flow(
            key,
            base_dist=dist.Normal(jnp.zeros(SIRSDESimulator.out_dim)),
            cond_dim=SIRSDESimulator.in_dim,
        )
        self.simulator = SimulatorDistribution(
            SIRSDESimulator(time_steps=40),
            shape=(SIRSDESimulator.out_dim,),
            cond_shape=(SIRSDESimulator.in_dim,),
            surrogate=surrogate,
        )

    def __call__(self, obs: Array | None = None):
        beta_mean = 0.1
        beta_count = 10
        alpha = beta_mean * beta_count
        beta = beta_count * (1 - beta_mean)
        recovery_rate = sample("recovery_rate", dist.Beta(alpha, beta))
        infection_rate = sample("infection_rate", dist.Beta(alpha, beta))
        r0_mean_reversion = sample("r0_mean_reversion", dist.Beta(alpha, beta))
        r0_volatility = sample("r0_volatility", dist.Beta(alpha, beta))

        z = deterministic(
            "z",
            jnp.stack(
                [recovery_rate, infection_rate, r0_mean_reversion, r0_volatility],
            ),
        )
        sample("x", self.simulator, condition=z, obs=obs)


class SIRSDEGuide(AbstractProgram):
    posterior: dist.Transformed

    def __init__(self, key: PRNGKeyArray):

        self.posterior = dist.Transformed(
            masked_autoregressive_flow(
                key,
                base_dist=dist.Normal(-jnp.ones(SIRSDESimulator.in_dim)),
                cond_dim=SIRSDESimulator.out_dim,
            ),
            bij.Sigmoid((4,)),  # Map to [0, 1] support
        )

    def __call__(self, obs: Array | None = None):
        z = sample("z", self.posterior, condition=obs)
        deterministic("recovery_rate", z[0])
        deterministic("infection_rate", z[1])
        deterministic("r0_mean_reversion", z[2])
        deterministic("r0_volatility", z[3])


class SIRSDESimulator(eqx.Module):
    """An Susceptible-Infected-Recovered epidemic model, with a stochastic R0."""

    time_steps: int
    max_solve_steps: int
    in_dim = 4
    out_dim = 3

    def __init__(self, *, time_steps: int, max_solve_steps: int = 5000):
        self.time_steps = time_steps
        self.max_solve_steps = max_solve_steps

    @eqx.filter_jit
    def __call__(
        self,
        key: PRNGKeyArray,
        condition: Array,
    ):
        key1, key2 = jr.split(key)
        simulation = self.simulate(key1, *condition)
        summaries = self.summarize(key2, simulation)
        assert summaries.shape == (self.out_dim,)
        return summaries

    @eqx.filter_jit
    def simulate(
        self,
        key,
        infection_rate: ScalarLike,
        recovery_rate: ScalarLike,
        r0_mean_reversion: ScalarLike,
        r0_volatility: ScalarLike,
    ):
        t0, t1 = 0, self.time_steps + 1
        ode = partial(
            self.ode,
            infection_rate=infection_rate,
            recovery_rate=recovery_rate,
            r0_mean_reversion=r0_mean_reversion,
        )
        sde = partial(self.sde, r0_volatility=r0_volatility)

        brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-2, shape=(), key=key)
        r0_init = infection_rate / recovery_rate
        sol = diffeqsolve(
            terms=MultiTerm(ODETerm(ode), ControlTerm(sde, brownian_motion)),
            solver=Euler(),
            t0=t0,
            t1=t1,
            dt0=0.01,
            y0=jnp.array([0.99, 0.01, 0, r0_init]),
            saveat=SaveAt(ts=range(1, self.time_steps + 1)),
            max_steps=self.max_solve_steps,
        )
        return sol.ys[:, 1]

    def ode(
        self,
        t,
        y,
        *args,
        infection_rate,
        recovery_rate,
        r0_mean_reversion,
    ):
        """ODE portion defined compatible with Diffrax."""
        s, i, r, r0 = y
        newly_infected = r0 * recovery_rate * s
        newly_recovered = recovery_rate * i
        ds = -newly_infected
        di = newly_infected - newly_recovered
        dr = newly_recovered
        dR0 = r0_mean_reversion * (infection_rate / recovery_rate - r0)
        return jnp.hstack((ds, di, dr, dR0))

    def sde(self, t, y, *args, r0_volatility):
        """SDE portion compatible with Diffrax.

        We scale the brownian motion by the square root of R0 to ensure positivity (i.e.
        the mean reversion will dominate).
        """
        scale = jnp.sqrt(jnp.abs(y[-1]))
        return scale * jnp.array([0, 0, 0, r0_volatility])

    def summarize(self, key: PRNGKeyArray, x: Array):
        x = jnp.clip(x, a_min=1e-5)
        max_ = x.max()
        max_at = (jnp.argmax(x) + jr.uniform(key)) / self.time_steps
        vol = jnp.std(jnp.diff(jnp.log(x)))
        return jnp.array([max_, max_at, vol])
