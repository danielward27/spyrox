"""A collection of tasks for validating model performance."""

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
from flowjax.distributions import LogNormal, Normal, Transformed
from flowjax.experimental.numpyro import sample
from flowjax.flows import masked_autoregressive_flow
from jax import Array
from jaxtyping import PRNGKeyArray, ScalarLike
from numpyro import deterministic
from numpyro.infer.reparam import TransformReparam
from pyrox.program import AbstractProgram, ReparameterizedProgram

from spyrox.simulator import AbstactProgramWithSurrogate, SimulatorToDistribution
from spyrox.utils import VmapDistribution

# TODO force setting of condition, prior to making dist?
# Then VmapDistribution, specifying condition used to add batch dim for amortized
# VI?


def get_task(key):
    key, subkey = jr.split(key)
    model = SIRSDEModel(subkey)
    key, subkey = jr.split(key)
    guide = SIRSDEGuide(subkey)  # Currently not used.
    latents = [
        "infection_rate_mean",
        "recovery_rate_mean",
        "r0_mean_reversion_mean",
        "r0_volatility_mean",
        "infection_rate",
        "recovery_rate",
        "r0_mean_reversion",
        "r0_volatility",
    ]
    reparam = {n: TransformReparam() for n in latents}
    model = ReparameterizedProgram(model, reparam)
    return model, guide


class SIRSDEModel(AbstactProgramWithSurrogate):
    surrogate: Transformed
    simulator: SimulatorToDistribution
    n_obs = 2
    local_param_dim = 4

    def __init__(
        self,
        key: PRNGKeyArray,
    ):
        self.surrogate = masked_autoregressive_flow(
            key,
            base_dist=dist.Normal(jnp.zeros(SIRSDESimulator.out_dim)),
            cond_dim=SIRSDESimulator.in_dim,
        )
        self.simulator = SimulatorToDistribution(
            SIRSDESimulator(time_steps=20),
            shape=(SIRSDESimulator.out_dim,),
            cond_shape=(SIRSDESimulator.in_dim,),
        )

    def __call__(self, *, use_surrogate: bool, obs: Array | None = None):
        infection_rate_mean = sample("infection_rate_mean", Normal(-1.5, 0.4))
        recovery_rate_mean = sample("recovery_rate_mean", Normal(-1.5, 0.4))
        r0_mean_reversion_mean = sample("r0_mean_reversion_mean", Normal(-1.5, 0.4))
        r0_volatility_mean = sample("r0_volatility_mean", Normal(-1.5, 0.4))

        with numpyro.plate("n_obs", SIRSDEModel.n_obs):
            infection_rate = sample(
                "infection_rate",
                LogNormal(infection_rate_mean, 0.2),
            )

            recovery_rate = sample(
                "recovery_rate",
                LogNormal(recovery_rate_mean, 0.2),
            )

            r0_mean_reversion = sample(
                "r0_mean_reversion",
                LogNormal(r0_mean_reversion_mean, 0.2),
            )
            r0_volatility = sample(
                "r0_volatility",
                LogNormal(r0_volatility_mean, 0.2),
            )
            z = jnp.stack(
                [infection_rate, recovery_rate, r0_mean_reversion, r0_volatility],
                axis=1,
            )
            z = deterministic("z", jnp.clip(z, min=1e-7, max=1 - 1e-7))

            if use_surrogate:
                sample("x", self.surrogate, condition=z, obs=obs)
            else:
                sample("x", self.simulator, condition=z, obs=obs)


class SIRSDEGuide(AbstractProgram):
    local_posterior: ndist.Normal
    global_posterior: ndist.Normal

    def __init__(self, key: PRNGKeyArray):
        global_param_dim = 4
        self.global_posterior = Normal(jnp.zeros(global_param_dim))
        self.local_posterior = eqx.filter_vmap(Normal)(
            jnp.zeros((SIRSDEModel.n_obs, SIRSDEModel.local_param_dim)),
        )

    def __call__(self, obs: Array | None = None):  # TODO obs ignored.
        global_params = sample("global_params", self.global_posterior)
        deterministic("infection_rate_mean_base", global_params[0])
        deterministic("recovery_rate_mean_base", global_params[1])
        deterministic("r0_mean_reversion_mean_base", global_params[2])
        deterministic("r0_volatility_mean_base", global_params[3])

        with numpyro.plate("n_obs", size=SIRSDEModel.n_obs):
            local_dist = VmapDistribution(self.local_posterior)
            z_base = sample("z_base", local_dist)

        deterministic("infection_rate_base", z_base[:, 0])
        deterministic("recovery_rate_base", z_base[:, 1])
        deterministic("r0_mean_reversion_base", z_base[:, 2])
        deterministic("r0_volatility_base", z_base[:, 3])


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
        return jnp.array([max_, max_at, vol])


# def get_task(key):
#     key, subkey = jr.split(key)
#     model = SIRSDEModel(subkey)
#     key, subkey = jr.split(key)
#     guide = SIRSDEGuide(subkey)  # Currently not used.
#     latents = [
#         "infection_rate_mean",
#         "recovery_rate_mean",
#         "r0_mean_reversion_mean",
#         "r0_volatility_mean",
#         "infection_rate",
#         "recovery_rate",
#         "r0_mean_reversion",
#         "r0_volatility",
#     ]
#     reparam = {n: TransformReparam() for n in latents}
#     model = ReparameterizedProgram(model, reparam)
#     return model, guide


# class SIRSDEModel(AbstactProgramWithSurrogate):
#     surrogate: Transformed
#     simulator: SimulatorToDistribution
#     n_obs = 2
#     local_param_dim = 4

#     def __init__(
#         self,
#         key: PRNGKeyArray,
#     ):
#         self.surrogate = block_neural_autoregressive_flow(
#             key,
#             base_dist=dist.Normal(jnp.zeros(SIRSDESimulator.out_dim)),
#             cond_dim=SIRSDESimulator.in_dim,
#         )
#         self.simulator = SimulatorToDistribution(
#             SIRSDESimulator(time_steps=20),
#             shape=(SIRSDESimulator.out_dim,),
#             cond_shape=(SIRSDESimulator.in_dim,),
#         )

#     def __call__(self, *, use_surrogate: bool, obs: Array | None = None):
#         infection_rate_mean = sample("infection_rate_mean", Normal(-1.5, 0.4))
#         recovery_rate_mean = sample("recovery_rate_mean", Normal(-1.5, 0.4))
#         r0_mean_reversion_mean = sample("r0_mean_reversion_mean", Normal(-1.5, 0.4))
#         r0_volatility_mean = sample("r0_volatility_mean", Normal(-1.5, 0.4))

#         numpyro.deterministic(
#             "global_params",
#             jnp.hstack(
#                 [
#                     infection_rate_mean,
#                     recovery_rate_mean,
#                     r0_mean_reversion_mean,
#                     r0_volatility_mean,
#                 ],
#             ),
#         )

#         with numpyro.plate("n_obs", SIRSDEModel.n_obs):
#             infection_rate = sample(
#                 "infection_rate",
#                 LogNormal(infection_rate_mean, 0.2),
#             )

#             recovery_rate = sample(
#                 "recovery_rate",
#                 LogNormal(recovery_rate_mean, 0.2),
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
#             # Both for numerical stability and to ensure valid simulator params
#             deterministic(
#                 "z",
#                 jnp.clip(z, min=1e-7, max=1 - 1e-7),
#             )  # TODO this doesn't align with below!

#             if use_surrogate:
#                 sample("x", self.surrogate, condition=z, obs=obs)
#             else:
#                 sample("x", self.simulator, condition=z, obs=obs)


# class SIRSDEGuide(AbstractProgram):
#     local_posterior: ndist.Normal
#     global_posterior: ndist.Normal

#     def __init__(self, key: PRNGKeyArray):
#         global_param_dim = 4
#         self.global_posterior = Normal(jnp.zeros(global_param_dim))
#         self.local_posterior = eqx.filter_vmap(Normal)(
#             jnp.zeros((SIRSDEModel.n_obs, SIRSDEModel.local_param_dim)),
#         )

#     def __call__(self, obs: Array | None = None):  # TODO obs ignored.
#         global_params = sample("global_params", self.global_posterior)
#         deterministic("infection_rate_mean_base", global_params[0])
#         deterministic("recovery_rate_mean_base", global_params[1])
#         deterministic("r0_mean_reversion_mean_base", global_params[2])
#         deterministic("r0_volatility_mean_base", global_params[3])

#         with numpyro.plate("n_obs", size=SIRSDEModel.n_obs):
#             local_dist = VmapDistribution(
#                 self.local_posterior,
#                 SIRSDEModel.n_obs,
#             )

#             z_base = sample("z_base", local_dist)
#             z = deterministic("z", )

#         deterministic("infection_rate_base", z_base[:, 0])
#         deterministic("recovery_rate_base", z_base[:, 1])
#         deterministic("r0_mean_reversion_base", z_base[:, 2])
#         deterministic("r0_volatility_base", z_base[:, 3])


# class SIRSDESimulator(eqx.Module):
#     """An Susceptible-Infected-Recovered epidemic model, with a stochastic R0."""

#     time_steps: int
#     max_solve_steps: int
#     in_dim = 4
#     out_dim = 20

#     def __init__(self, *, time_steps: int, max_solve_steps: int = 5000):
#         self.time_steps = time_steps
#         self.max_solve_steps = max_solve_steps

#     @eqx.filter_jit
#     def __call__(
#         self,
#         key: PRNGKeyArray,
#         condition: Array,
#     ):
#         key1, key2 = jr.split(key)
#         simulation = self.simulate(key1, *condition)
#         summaries = self.summarize(key2, simulation)
#         assert summaries.shape == (self.out_dim,)
#         return summaries

#     @eqx.filter_jit
#     def simulate(
#         self,
#         key,
#         infection_rate: ScalarLike,
#         recovery_rate: ScalarLike,
#         r0_mean_reversion: ScalarLike,
#         r0_volatility: ScalarLike,
#     ):
#         t0, t1 = 0, self.time_steps + 1
#         ode = partial(
#             self.ode,
#             infection_rate=infection_rate,
#             recovery_rate=recovery_rate,
#             r0_mean_reversion=r0_mean_reversion,
#         )
#         sde = partial(self.sde, r0_volatility=r0_volatility)

#         brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-2, shape=(), key=key)
#         r0_init = infection_rate / recovery_rate
#         sol = diffeqsolve(
#             terms=MultiTerm(ODETerm(ode), ControlTerm(sde, brownian_motion)),
#             solver=Euler(),
#             t0=t0,
#             t1=t1,
#             dt0=0.01,
#             y0=jnp.array([0.99, 0.01, 0, r0_init]),
#             saveat=SaveAt(ts=range(1, self.time_steps + 1)),
#             max_steps=self.max_solve_steps,
#         )

#         return jnp.nan_to_num(sol.ys[:, 1], nan=0, posinf=0, neginf=0)

#     def ode(
#         self,
#         t,
#         y,
#         *args,
#         infection_rate,
#         recovery_rate,
#         r0_mean_reversion,
#     ):
#         """ODE portion defined compatible with Diffrax."""
#         s, i, r, r0 = y
#         newly_infected = r0 * recovery_rate * s
#         newly_recovered = recovery_rate * i
#         ds = -newly_infected
#         di = newly_infected - newly_recovered
#         dr = newly_recovered
#         dR0 = r0_mean_reversion * (infection_rate / recovery_rate - r0)
#         return jnp.hstack((ds, di, dr, dR0))

#     def sde(self, t, y, *args, r0_volatility):
#         """SDE portion compatible with Diffrax.

#         We scale the brownian motion by the square root of R0 to ensure positivity (i.e.
#         the mean reversion will dominate).
#         """
#         scale = jnp.sqrt(jnp.abs(y[-1]))
#         return scale * jnp.array([0, 0, 0, r0_volatility])

#     def summarize(self, key: PRNGKeyArray, x: Array):
#         return x
#         # x = jnp.clip(x, a_min=1e-5)
#         # max_ = x.max()
#         # max_at = (jnp.argmax(x) + jr.uniform(key)) / self.time_steps
#         # vol = jnp.std(jnp.diff(jnp.log(x)))
#         # return jnp.array([max_, max_at, vol])


# %%
# batch_size = 5
# vmap_normal = VmapDistribution(
#     eqx.filter_vmap(Normal)(jnp.zeros((5, 3))),
#     batch_size=SIRSDEModel.n_obs,
# )
# samp = vmap_normal.sample(jr.key(0), (2,))

# assert samp.shape == (2, 5, 3)
# assert vmap_normal.log_prob(samp).shape == (2, 5)
# %%

# """A collection of tasks for validating model performance."""

# # %%
# from functools import partial

# import equinox as eqx
# import flowjax.distributions as dist
# import jax.numpy as jnp
# import jax.random as jr
# import numpyro
# from diffrax import (
#     ControlTerm,
#     Euler,
#     MultiTerm,
#     ODETerm,
#     SaveAt,
#     VirtualBrownianTree,
#     diffeqsolve,
# )
# from flowjax import bijections as bij
# from flowjax.distributions import Transformed
# from flowjax.experimental.numpyro import sample
# from flowjax.flows import block_neural_autoregressive_flow, coupling_flow
# from jax import Array, vmap
# from jaxtyping import PRNGKeyArray, ScalarLike
# from numpyro import deterministic
# from pyrox.program import AbstractProgram

# from spyrox.simulator import AbstactProgramWithSurrogate, SimulatorToDistribution


# class SIRSDEModel(AbstactProgramWithSurrogate):
#     surrogate: Transformed
#     simulator: SimulatorToDistribution
#     n_obs = 5

#     def __init__(
#         self,
#         key: PRNGKeyArray,
#     ):
#         self.surrogate = block_neural_autoregressive_flow(
#             key,
#             base_dist=dist.Normal(jnp.zeros(SIRSDESimulator.out_dim)),
#             cond_dim=SIRSDESimulator.in_dim,
#         )
#         self.simulator = SimulatorToDistribution(
#             SIRSDESimulator(time_steps=20),
#             shape=(SIRSDESimulator.out_dim,),
#             cond_shape=(SIRSDESimulator.in_dim,),
#         )

#     def __call__(self, *, use_surrogate: bool, obs: Array | None = None):
#         beta_count = 25  # Higher will be tighter

#         # Give global betas a mean of 0.1
#         a = 0.1 * beta_count
#         b = (1 - 0.1) * beta_count

#         infection_rate_mean = sample("infection_rate_mean", dist.Beta(a, b))
#         recovery_rate_mean = sample("recovery_rate_mean", dist.Beta(a, b))
#         r0_mean_reversion_mean = sample("r0_mean_reversion_mean", dist.Beta(a, b))
#         r0_volatility_mean = sample("r0_volatility_mean", dist.Beta(a, b))

#         # TODO consider setting plate dim to one for simulating?
#         # TODO can we avoid sampling global on round 1+?
#         numpyro.deterministic(
#             "global_params",
#             jnp.hstack(
#                 [
#                     infection_rate_mean,
#                     recovery_rate_mean,
#                     r0_mean_reversion_mean,
#                     r0_volatility_mean,
#                 ],
#             ),
#         )

#         with numpyro.plate("n_obs", SIRSDEModel.n_obs):
#             infection_rate = sample(
#                 "infection_rate",
#                 dist.Beta(
#                     infection_rate_mean * beta_count,
#                     beta_count * (1 - infection_rate_mean),
#                 ),
#             )

#             recovery_rate = sample(
#                 "recovery_rate",
#                 dist.Beta(
#                     recovery_rate_mean * beta_count,
#                     beta_count * (1 - recovery_rate_mean),
#                 ),
#             )

#             r0_mean_reversion = sample(
#                 "r0_mean_reversion",
#                 dist.Beta(
#                     r0_mean_reversion_mean * beta_count,
#                     beta_count * (1 - r0_mean_reversion_mean),
#                 ),
#             )
#             r0_volatility = sample(
#                 "r0_volatility",
#                 dist.Beta(
#                     r0_volatility_mean * beta_count,
#                     beta_count * (1 - r0_volatility_mean),
#                 ),
#             )
#             z = jnp.stack(
#                 [infection_rate, recovery_rate, r0_mean_reversion, r0_volatility],
#                 axis=1,
#             )
#             # For numerical stability, we clip to avoid inv(sigmoid)(0)
#             deterministic("z", jnp.clip(z, min=1e-7))

#             if use_surrogate:
#                 sample("x", self.surrogate, condition=z, obs=obs)
#             else:
#                 sample("x", self.simulator, condition=z, obs=obs)


# class SIRSDEGuide(AbstractProgram):
#     local_posterior: dist.Transformed
#     global_posterior: dist.Transformed

#     def __init__(self, key: PRNGKeyArray):
#         global_param_dim = 4
#         local_param_dim = 4

#         key, subkey = jr.split(key)
#         # q(tau|tau, \mathcal{x})
#         self.global_posterior = dist.Transformed(
#             coupling_flow(
#                 subkey,
#                 base_dist=dist.Normal(-jnp.ones(global_param_dim)),
#                 flow_layers=4,
#             ),
#             bij.Sigmoid((4,)),  # Map to [0, 1] support
#         )

#         key, subkey = jr.split(key)
#         self.local_posterior = dist.Transformed(
#             coupling_flow(
#                 subkey,
#                 base_dist=dist.Normal(-jnp.ones(local_param_dim)),
#                 cond_dim=SIRSDESimulator.out_dim + global_param_dim,
#                 flow_layers=8,
#             ),
#             bij.Sigmoid((local_param_dim,)),  # Map to [0, 1] support
#         )

#     def __call__(self, *, obs: Array):
#         global_params = sample("global_params", self.global_posterior)
#         deterministic("infection_rate_mean", global_params[0])
#         deterministic("recovery_rate_mean", global_params[1])
#         deterministic("r0_mean_reversion_mean", global_params[2])
#         deterministic("r0_volatility_mean", global_params[3])

#         local_condition = jnp.hstack(
#             (obs, jnp.tile(global_params, (SIRSDEModel.n_obs, 1))),
#         )

#         with numpyro.plate("n_obs", 5):
#             z = sample("z", self.local_posterior, condition=local_condition)

#         deterministic("infection_rate", z[:, 0])
#         deterministic("recovery_rate", z[:, 1])
#         deterministic("r0_mean_reversion", z[:, 2])
#         deterministic("r0_volatility", z[:, 3])


# class SIRSDESimulator(eqx.Module):
#     """An Susceptible-Infected-Recovered epidemic model, with a stochastic R0."""

#     time_steps: int
#     max_solve_steps: int
#     in_dim = 4
#     out_dim = 20

#     def __init__(self, *, time_steps: int, max_solve_steps: int = 5000):
#         self.time_steps = time_steps
#         self.max_solve_steps = max_solve_steps

#     @eqx.filter_jit
#     def __call__(
#         self,
#         key: PRNGKeyArray,
#         condition: Array,
#     ):
#         key1, key2 = jr.split(key)
#         simulation = self.simulate(key1, *condition)
#         summaries = self.summarize(key2, simulation)
#         assert summaries.shape == (self.out_dim,)
#         return summaries

#     @eqx.filter_jit
#     def simulate(
#         self,
#         key,
#         infection_rate: ScalarLike,
#         recovery_rate: ScalarLike,
#         r0_mean_reversion: ScalarLike,
#         r0_volatility: ScalarLike,
#     ):
#         t0, t1 = 0, self.time_steps + 1
#         ode = partial(
#             self.ode,
#             infection_rate=infection_rate,
#             recovery_rate=recovery_rate,
#             r0_mean_reversion=r0_mean_reversion,
#         )
#         sde = partial(self.sde, r0_volatility=r0_volatility)

#         brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-2, shape=(), key=key)
#         r0_init = infection_rate / recovery_rate
#         sol = diffeqsolve(
#             terms=MultiTerm(ODETerm(ode), ControlTerm(sde, brownian_motion)),
#             solver=Euler(),
#             t0=t0,
#             t1=t1,
#             dt0=0.01,
#             y0=jnp.array([0.99, 0.01, 0, r0_init]),
#             saveat=SaveAt(ts=range(1, self.time_steps + 1)),
#             max_steps=self.max_solve_steps,
#         )

#         return jnp.nan_to_num(sol.ys[:, 1], nan=0, posinf=0, neginf=0)

#     def ode(
#         self,
#         t,
#         y,
#         *args,
#         infection_rate,
#         recovery_rate,
#         r0_mean_reversion,
#     ):
#         """ODE portion defined compatible with Diffrax."""
#         s, i, r, r0 = y
#         newly_infected = r0 * recovery_rate * s
#         newly_recovered = recovery_rate * i
#         ds = -newly_infected
#         di = newly_infected - newly_recovered
#         dr = newly_recovered
#         dR0 = r0_mean_reversion * (infection_rate / recovery_rate - r0)
#         return jnp.hstack((ds, di, dr, dR0))

#     def sde(self, t, y, *args, r0_volatility):
#         """SDE portion compatible with Diffrax.

#         We scale the brownian motion by the square root of R0 to ensure positivity (i.e.
#         the mean reversion will dominate).
#         """
#         scale = jnp.sqrt(jnp.abs(y[-1]))
#         return scale * jnp.array([0, 0, 0, r0_volatility])

#     def summarize(self, key: PRNGKeyArray, x: Array):
#         return x
#         # x = jnp.clip(x, a_min=1e-5)
#         # max_ = x.max()
#         # max_at = (jnp.argmax(x) + jr.uniform(key)) / self.time_steps
#         # vol = jnp.std(jnp.diff(jnp.log(x)))
#         # return jnp.array([max_, max_at, vol])

# %%
