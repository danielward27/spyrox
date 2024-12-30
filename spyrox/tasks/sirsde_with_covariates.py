import equinox as eqx
import flowjax.distributions as dist
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as ndist
from flowjax.distributions import AbstractDistribution, LogNormal, Normal, Transformed
from flowjax.experimental.numpyro import sample
from flowjax.flows import masked_autoregressive_flow
from jax import Array
from jaxtyping import PRNGKeyArray
from numpyro import deterministic
from numpyro.infer.reparam import TransformReparam
from pyrox.program import AbstractProgram, ReparameterizedProgram

from spyrox.simulator import AbstactProgramWithSurrogate, SimulatorToDistribution
from spyrox.tasks.sirsde import SIRSDESimulator
from spyrox.utils import VmapDistribution


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


from flowjax.bijections import RationalQuadraticSpline
from flowjax.flows import masked_autoregressive_flow


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
                flow_layers=4,
                transformer=RationalQuadraticSpline(knots=8, interval=4),
            )

        self.z_dist = eqx.filter_vmap(get_flow)(jr.split(key, n_obs))

    def __call__(self, obs: Array | None = None):  # TODO obs ignored.
        sample("infection_rate_beta_base", self.infection_rate_beta_dist)
        sample("infection_rate_bias_base", self.infection_rate_bias_dist)

        sample("recovery_rate_beta_base", self.recovery_rate_beta_dist)
        sample("recovery_rate_bias_base", self.recovery_rate_bias_dist)

        sample("r0_mean_reversion_mean_base", self.r0_mean_reversion_mean_dist)
        sample("r0_volatility_mean_base", self.r0_volatility_mean_dist)

        with numpyro.plate("n_obs", size=SIRSDECovariateModel.n_obs):
            sample("z_base", VmapDistribution(self.z_dist))


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
