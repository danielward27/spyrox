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
from flowjax.flows import masked_autoregressive_flow
from jax import Array
from jaxtyping import PRNGKeyArray, ScalarLike
from numpyro.infer.reparam import TransformReparam
from pyrox.program import AbstractProgram, ReparameterizedProgram

from spyrox.simulator import AbstactProgramWithSurrogate, SimulatorToDistribution
from spyrox.tasks.sirsde import SIRSDESimulator
from spyrox.utils import VmapDistribution


class _SetCondition(AbstractDistribution):
    # TODO document. Make sure condition isn't trained if used.
    dist: AbstractDistribution
    condition: Array
    shape: tuple[int, ...]
    cond_shape = None

    def __init__(self, dist, condition):
        self.condition = condition
        self.shape = dist.shape
        self.dist = dist

    def _sample(self, key, condition=None):
        return self.dist._sample(key, condition=self.condition)

    def _sample_and_log_prob(self, key, condition=None):
        return self.dist._sample_and_log_prob(key, condition=self.condition)

    def _log_prob(self, x, condition=None):
        return self.dist._log_prob(x, condition=self.condition)


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
        "infection_rate_beta",
        "infection_rate_bias",
        "recovery_rate_beta",
        "recovery_rate_bias",
        "r0_mean_reversion_mean",
        "r0_volatility_mean",
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
        vol = jnp.std(jnp.diff(jnp.log(x)))

        # Standardise with location and scale from seperate set of simulations
        s_loc = jnp.array([0.38, 0.21, 0.14])
        s_scale = jnp.array([0.14, 0.12, 0.03])
        s = jnp.array([max_, max_at, vol])
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
            flow_layers=3,
            transformer=RationalQuadraticSpline(knots=5, interval=4),
        )

        self.simulator = SimulatorToDistribution(
            SIRSDESimulator(time_steps=20),
            shape=(SIRSDESimulator.out_dim,),
            cond_shape=(SIRSDESimulator.in_dim,),
        )
        assert covariates.shape == (self.n_obs, self.n_covariates)
        self.covariates = covariates

    def __call__(self, *, use_surrogate: bool, obs: Array | None = None):
        infection_rate_beta = sample(
            "infection_rate_beta",
            Normal(jnp.zeros(self.covariates.shape[1]), 0.25),
        )

        infection_rate_bias = sample(
            "infection_rate_bias",
            Normal(-1.5, 0.4),
        )

        recovery_rate_beta = sample(
            "recovery_rate_beta",
            Normal(jnp.zeros(self.covariates.shape[1]), 0.25),
        )

        recovery_rate_bias = sample(
            "recovery_rate_bias",
            Normal(-1.5, 0.4),
        )

        r0_mean_reversion_mean = sample("r0_mean_reversion_mean", Normal(-1.5, 0.4))
        r0_volatility_mean = sample("r0_volatility_mean", Normal(-2, 0.4))

        infection_rate_means = (
            self.covariates @ infection_rate_beta + infection_rate_bias
        )
        recovery_rate_means = self.covariates @ recovery_rate_beta + recovery_rate_bias

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


class SIRSDECovariateGuide(AbstractProgram):
    infection_rate_beta_dist: Normal
    infection_rate_bias_dist: Normal
    recovery_rate_beta_dist: Normal
    recovery_rate_bias_dist: Normal
    r0_mean_reversion_mean_dist: Normal
    r0_volatility_mean_dist: Normal
    z_dist: AbstractDistribution

    def __init__(self, key: PRNGKeyArray):
        n_cov = SIRSDECovariateModel.n_covariates
        n_obs = SIRSDECovariateModel.n_obs
        self.infection_rate_beta_dist = Normal(jnp.zeros(n_cov))
        self.infection_rate_bias_dist = Normal()
        self.recovery_rate_beta_dist = Normal(jnp.zeros(n_cov))
        self.recovery_rate_bias_dist = Normal()
        self.r0_mean_reversion_mean_dist = Normal()
        self.r0_volatility_mean_dist = Normal()

        def get_flow(key):
            return masked_autoregressive_flow(
                key,
                base_dist=Normal(jnp.zeros(SIRSDECovariateModel.n_z)),
                flow_layers=3,
                transformer=RationalQuadraticSpline(knots=5, interval=4),
                cond_dim=10,
            )

        self.z_dist = eqx.filter_vmap(get_flow)(jr.split(key, n_obs))

    def __call__(self, obs: Array | None = None):  # TODO obs ignored.
        global_params = [
            sample("infection_rate_beta_base", self.infection_rate_beta_dist),
            sample("infection_rate_bias_base", self.infection_rate_bias_dist),
            sample("recovery_rate_beta_base", self.recovery_rate_beta_dist),
            sample("recovery_rate_bias_base", self.recovery_rate_bias_dist),
            sample("r0_mean_reversion_mean_base", self.r0_mean_reversion_mean_dist),
            sample("r0_volatility_mean_base", self.r0_volatility_mean_dist),
        ]
        global_stacked = jnp.concatenate(
            [jnp.atleast_1d(p) for p in global_params],
        )
        global_stacked = jnp.tile(global_stacked, SIRSDECovariateModel.n_obs).reshape(
            SIRSDECovariateModel.n_obs,
            -1,
        )
        # Repeat for each obs

        with numpyro.plate("n_obs", size=SIRSDECovariateModel.n_obs):
            sample(
                "z_base",
                VmapDistribution(_SetCondition(self.z_dist, global_stacked)),
            )


# class SIRSDECovariateGuide(AbstractProgram):
#     infection_rate_beta_dist: Normal
#     infection_rate_bias_dist: Normal
#     recovery_rate_beta_dist: Normal
#     recovery_rate_bias_dist: Normal
#     r0_mean_reversion_mean_dist: Normal
#     r0_volatility_mean_dist: Normal
#     z_dist: AbstractDistribution

#     def __init__(self, key: PRNGKeyArray):
#         n_cov = SIRSDECovariateModel.n_covariates
#         n_obs = SIRSDECovariateModel.n_obs
#         self.infection_rate_beta_dist = Normal(jnp.zeros(n_cov))
#         self.infection_rate_bias_dist = Normal()
#         self.recovery_rate_beta_dist = Normal(jnp.zeros(n_cov))
#         self.recovery_rate_bias_dist = Normal()
#         self.r0_mean_reversion_mean_dist = Normal()
#         self.r0_volatility_mean_dist = Normal()

#         def get_flow(key):
#             return masked_autoregressive_flow(
#                 key,
#                 base_dist=Normal(jnp.zeros(SIRSDECovariateModel.n_z)),
#                 flow_layers=3,
#                 transformer=RationalQuadraticSpline(knots=5, interval=4),
#             )

#         self.z_dist = eqx.filter_vmap(get_flow)(jr.split(key, n_obs))

#     def __call__(self, obs: Array | None = None):  # TODO obs ignored.
#         global_params = [
#             sample("infection_rate_beta_base", self.infection_rate_beta_dist),
#             sample("infection_rate_bias_base", self.infection_rate_bias_dist),
#             sample("recovery_rate_beta_base", self.recovery_rate_beta_dist),
#             sample("recovery_rate_bias_base", self.recovery_rate_bias_dist),
#             sample("r0_mean_reversion_mean_base", self.r0_mean_reversion_mean_dist),
#             sample("r0_volatility_mean_base", self.r0_volatility_mean_dist),
#         ]

#         with numpyro.plate("n_obs", size=SIRSDECovariateModel.n_obs):
#             sample("z_base", VmapDistribution(self.z_dist))


# class SIRSDECovariateModel(AbstactProgramWithSurrogate):
#     surrogate: Transformed
#     simulator: SimulatorToDistribution
#     local_param_dim = 4
#     covariates: Array
#     n_obs = 20
#     n_covariates = 3

#     def __init__(
#         self,
#         key: PRNGKeyArray,
#         covariates: Array,
#     ):
#         self.surrogate = masked_autoregressive_flow(
#             key,
#             base_dist=dist.Normal(jnp.zeros(SIRSDESimulator.out_dim)),
#             cond_dim=SIRSDESimulator.in_dim,
#         )
#         self.simulator = SimulatorToDistribution(
#             SIRSDESimulator(time_steps=20),
#             shape=(SIRSDESimulator.out_dim,),
#             cond_shape=(SIRSDESimulator.in_dim,),
#         )
#         assert covariates.shape == (self.n_obs, self.n_covariates)
#         self.covariates = covariates

#     def __call__(self, *, use_surrogate: bool, obs: Array | None = None):
#         infection_rate_beta = sample(
#             "infection_rate_beta",
#             Normal(jnp.zeros(self.covariates.shape[1]), 0.25),
#         )

#         infection_rate_bias = sample(
#             "infection_rate_bias",
#             Normal(-1.5, 0.4),
#         )

#         recovery_rate_beta = sample(
#             "recovery_rate_beta",
#             Normal(jnp.zeros(self.covariates.shape[1]), 0.25),
#         )

#         recovery_rate_bias = sample(
#             "recovery_rate_bias",
#             Normal(-1.5, 0.4),
#         )

#         r0_mean_reversion_mean = sample("r0_mean_reversion_mean", Normal(-1.5, 0.4))
#         r0_volatility_mean = sample("r0_volatility_mean", Normal(-2, 0.4))

#         infection_rate_means = (
#             self.covariates @ infection_rate_beta + infection_rate_bias
#         )
#         recovery_rate_means = self.covariates @ recovery_rate_beta + recovery_rate_bias

#         with numpyro.plate("n_obs", SIRSDECovariateModel.n_obs):

#             infection_rate = sample(
#                 "infection_rate",
#                 ndist.TransformedDistribution(
#                     ndist.Normal(jnp.zeros(self.n_obs), 1),
#                     [
#                         ndist.transforms.AffineTransform(infection_rate_means, 0.2),
#                         ndist.transforms.ExpTransform(),
#                     ],
#                 ),
#             )

#             recovery_rate = sample(
#                 "recovery_rate",
#                 ndist.TransformedDistribution(
#                     ndist.Normal(jnp.zeros(self.n_obs), 1),
#                     [
#                         ndist.transforms.AffineTransform(recovery_rate_means, 0.2),
#                         ndist.transforms.ExpTransform(),
#                     ],
#                 ),
#             )

#             r0_mean_reversion = sample(
#                 "r0_mean_reversion",
#                 LogNormal(r0_mean_reversion_mean, 0.2),
#             )

#             r0_volatility = sample(
#                 "r0_volatility",
#                 LogNormal(r0_volatility_mean, 0.2),
#             )
#             z = jnp.stack(
#                 [infection_rate, recovery_rate, r0_mean_reversion, r0_volatility],
#                 axis=1,
#             )
#             z = deterministic("z", jnp.clip(z, min=1e-7, max=1 - 1e-7))

#             if use_surrogate:
#                 sample("x", self.surrogate, condition=z, obs=obs)
#             else:
#                 sample("x", self.simulator, condition=z, obs=obs)


# class SIRSDECovariateGuide(AbstractProgram):
#     infection_rate_beta_dist: AbstractDistribution
#     infection_rate_bias_dist: AbstractDistribution
#     recovery_rate_beta_dist: AbstractDistribution
#     recovery_rate_bias_dist: AbstractDistribution
#     r0_mean_reversion_mean_dist: AbstractDistribution
#     r0_volatility_mean_dist: AbstractDistribution
#     infection_rate_dist: AbstractDistribution
#     recovery_rate_dist: AbstractDistribution
#     r0_mean_reversion_dist: AbstractDistribution
#     r0_volatility_dist: AbstractDistribution

#     def __init__(self, key: PRNGKeyArray):
#         n_cov = SIRSDECovariateModel.n_covariates
#         n_obs = SIRSDECovariateModel.n_obs
#         self.infection_rate_beta_dist = Normal(jnp.zeros(n_cov))
#         self.infection_rate_bias_dist = Normal()
#         self.recovery_rate_beta_dist = Normal(jnp.zeros(n_cov))
#         self.recovery_rate_bias_dist = Normal()
#         self.r0_mean_reversion_mean_dist = Normal()
#         self.r0_volatility_mean_dist = Normal()
#         self.infection_rate_dist = eqx.filter_vmap(Normal)(jnp.zeros(n_obs))
#         self.recovery_rate_dist = eqx.filter_vmap(Normal)(jnp.zeros(n_obs))
#         self.r0_mean_reversion_dist = eqx.filter_vmap(Normal)(jnp.zeros(n_obs))
#         self.r0_volatility_dist = eqx.filter_vmap(Normal)(jnp.zeros(n_obs))

#     def __call__(self, obs: Array | None = None):  # TODO obs ignored.
#         sample("infection_rate_beta_base", self.infection_rate_beta_dist)
#         sample("infection_rate_bias_base", self.infection_rate_bias_dist)

#         sample("recovery_rate_beta_base", self.recovery_rate_beta_dist)
#         sample("recovery_rate_bias_base", self.recovery_rate_bias_dist)

#         sample("r0_mean_reversion_mean_base", self.r0_mean_reversion_mean_dist)
#         sample("r0_volatility_mean_base", self.r0_volatility_mean_dist)

#         with numpyro.plate("n_obs", size=SIRSDECovariateModel.n_obs):
#             sample("infection_rate_base", VmapDistribution(self.infection_rate_dist))
#             sample("recovery_rate_base", VmapDistribution(self.recovery_rate_dist))
#             sample(
#                 "r0_mean_reversion_base",
#                 VmapDistribution(self.r0_mean_reversion_dist),
#             )
#             sample("r0_volatility_base", VmapDistribution(self.r0_volatility_dist))
