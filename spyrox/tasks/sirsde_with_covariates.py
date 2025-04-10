from functools import partial

import equinox as eqx
import flowjax.distributions as dist
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as ndist
from diffrax import (
    ControlTerm,
    Euler,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import AbstractDistribution, Normal, Transformed
from flowjax.experimental.numpyro import sample
from flowjax.flows import coupling_flow, masked_autoregressive_flow
from jax import Array
from jaxtyping import PRNGKeyArray, ScalarLike
from numpyro.infer.reparam import TransformReparam
from pyrox.program import AbstractProgram, ReparameterizedProgram

from spyrox.simulator import AbstactProgramWithSurrogate, SimulatorToDistribution
from spyrox.utils import VmapDistribution, _SetCondition


def get_task(
    key,
):
    key, subkey = jr.split(key)
    covariates = jr.normal(
        subkey,
        (SIRSDECovariateModel.n_obs, SIRSDECovariateModel.n_covariates),
    )

    model = SIRSDECovariateModel(subkey, covariates)
    key, subkey = jr.split(key)
    guide = SIRSDECovariateGuide(subkey)
    latents = [
        "tau",
        "z",
    ]
    reparam = {n: TransformReparam() for n in latents}
    model = ReparameterizedProgram(model, reparam)
    return model, guide


class SIRSDESimulator(eqx.Module):
    """An Susceptible-Infected-Recovered epidemic model, with a stochastic R0."""

    time_steps: int
    max_solve_steps: int
    in_dim = 4
    out_dim = 4

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
        infection_rate, recovery_rate, r0_mean_reversion, r0_volatility = (
            jnp.clip(p, min=1e-7, max=1 - 1e-7)
            for p in (infection_rate, recovery_rate, r0_mean_reversion, r0_volatility)
        )

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

        return jnp.nan_to_num(sol.ys[:, 1], nan=0, posinf=0, neginf=0)

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
        # return x
        x = jnp.clip(x, a_min=1e-5)
        max_ = x.max()
        max_at = (jnp.argmax(x) + jr.uniform(key)) / self.time_steps
        mean = jnp.mean(x)
        vol = jnp.std(jnp.diff(jnp.log(x)))

        # Standardise with location and scale from seperate set of simulations
        s_loc = jnp.array([0.38, 0.21, 0.22, 0.14])
        s_scale = jnp.array([0.14, 0.12, 0.1, 0.03])
        s = jnp.array([max_, max_at, mean, vol])
        return (s - s_loc) / s_scale


class SIRSDECovariateModel(AbstactProgramWithSurrogate):
    surrogate: Transformed
    simulator: SimulatorToDistribution
    covariates: Array
    n_obs = 10
    n_covariates = 3
    n_z = 4

    def __init__(
        self,
        key: PRNGKeyArray,
        covariates: Array,
    ):
        self.surrogate = masked_autoregressive_flow(
            key,
            base_dist=dist.Normal(jnp.zeros(SIRSDESimulator.out_dim)),
            cond_dim=SIRSDESimulator.in_dim,
            flow_layers=6,
            transformer=RationalQuadraticSpline(knots=8, interval=4),
        )

        self.simulator = SimulatorToDistribution(
            SIRSDESimulator(time_steps=20),
            shape=(SIRSDESimulator.out_dim,),
            cond_shape=(SIRSDESimulator.in_dim,),
        )
        assert covariates.shape == (self.n_obs, self.n_covariates)
        self.covariates = covariates

    def __call__(self, *, use_surrogate: bool, obs: Array | None = None):

        tau_dist = Normal(
            jnp.array([0, 0, 0, -1.5, 0, 0, 0, -1.5, -1.5, -2]),
            jnp.array([0.25, 0.25, 0.25, 0.4, 0.25, 0.25, 0.25, 0.4, 0.4, 0.4]),
        )
        tau = sample("tau", tau_dist)

        infection_rate_weights = tau[:3]
        infection_rate_bias = tau[3]
        recovery_rate_weights = tau[4:7]
        recovery_rate_bias = tau[7]
        r0_mean_reversion_mean = tau[8]
        r0_volatility_mean = tau[9]

        infection_rate_means = (
            self.covariates @ infection_rate_weights + infection_rate_bias
        )
        recovery_rate_means = (
            self.covariates @ recovery_rate_weights + recovery_rate_bias
        )

        z_means = jnp.stack(
            [
                infection_rate_means,
                recovery_rate_means,
                jnp.full(self.n_obs, r0_mean_reversion_mean),
                jnp.full(self.n_obs, r0_volatility_mean),
            ],
            axis=1,
        )

        with numpyro.plate("n_obs", SIRSDECovariateModel.n_obs):

            z = sample(
                "z",
                ndist.TransformedDistribution(
                    ndist.Normal(jnp.zeros_like(z_means), 1),
                    [
                        ndist.transforms.AffineTransform(z_means, 0.2),
                        ndist.transforms.ExpTransform(),
                    ],
                ).to_event(1),
            )

            if use_surrogate:
                sample("x", self.surrogate, condition=z, obs=obs)
            else:
                sample("x", self.simulator, condition=z, obs=obs)


import equinox as eqx
import jax


def scale_nn_initialisation(model, scale=0.01):
    # Encourage Gaussian global at init

    def map_fn(leaf):
        if isinstance(leaf, eqx.nn.Linear):
            return jax.tree.map(lambda x: scale * x, leaf)
        return leaf

    return jax.tree.map(
        map_fn,
        model,
        is_leaf=lambda leaf: isinstance(leaf, eqx.nn.Linear),
    )


from flowjax.distributions import MultivariateNormal
from paramax import WeightNormalization


class SIRSDECovariateGuide(AbstractProgram):
    tau_dist: Normal
    z_dist: AbstractDistribution

    def __init__(self, key: PRNGKeyArray):
        n_obs = SIRSDECovariateModel.n_obs
        tau_key, z_key = jr.split(key)

        # MVN + weight norm for better optimization
        tau_dist = MultivariateNormal(jnp.zeros(10), jnp.eye(10))
        self.tau_dist = eqx.tree_at(
            lambda dist: dist.bijection.triangular,
            tau_dist,
            replace_fn=WeightNormalization,
        )

        def get_flow(key):
            return coupling_flow(
                key,
                base_dist=Normal(jnp.zeros(SIRSDECovariateModel.n_z)),
                cond_dim=10,
                flow_layers=4,
                transformer=RationalQuadraticSpline(knots=6, interval=4),
            )

        self.z_dist = eqx.filter_vmap(get_flow)(jr.split(z_key, n_obs))

        # For amortized, we need following three lines:
        # 1) self.z_dist = masked_autoregressive_flow(*args, **kwargs, cond_dim=14)
        # 2) condition = jnp.concatenate([global_stacked, obs], axis=1)
        # 3) sample("z_base", self.z_dist, condition=condition)

    def __call__(self, obs: Array | None = None):  # TODO obs ignored.

        global_stacked = sample("tau_base", self.tau_dist)

        global_stacked = jnp.tile(global_stacked, SIRSDECovariateModel.n_obs).reshape(
            SIRSDECovariateModel.n_obs,
            -1,
        )

        with numpyro.plate("n_obs", size=SIRSDECovariateModel.n_obs):
            sample(
                "z_base",
                VmapDistribution(_SetCondition(self.z_dist, global_stacked)),
            )
