from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.train import fit_to_data, fit_to_variational_target
from jaxtyping import Array, PRNGKeyArray
from pyrox.program import AbstractProgram


def rounds_based(
    key: PRNGKeyArray,
    model: AbstractProgram,
    guide: AbstractProgram,
    num_rounds: int,
    sim_per_round: int,
    obs: Array,
    surrogate_fit_kwargs: dict,
    guide_fit_kwargs: dict,
    simulator_param_name: str,
    obs_name: str = "obs",
):

    losses = {
        "surrogate": {"train": [], "val": []},
        "guide": [],
    }

    for r in range(num_rounds):
        key, subkey = jr.split(key)
        if r == 0:
            latents = jax.vmap(model.sample)(jr.split(subkey, sim_per_round))
            simulations = latents.pop(obs_name)
        else:
            latents = jax.vmap(partial(guide.sample, obs=obs))(
                jr.split(subkey, sim_per_round),
            )
            key, subkey = jr.split(key)
            simulations = eqx.filter_vmap(model.sample)(
                jr.split(subkey, sim_per_round),
                latents,
            )[obs_name]

        # Fit simulator likelihood
        key, subkey = jr.split(key)
        surrogate, loss_vals = fit_to_data(
            subkey,
            dist=model.simulator.surrogate,
            x=simulations,
            condition=latents[simulator_param_name],
            **surrogate_fit_kwargs,
        )
        losses["surrogate"]["train"].append(loss_vals["train"])
        losses["surrogate"]["val"].append(loss_vals["val"])

        model = eqx.tree_at(
            where=lambda model: model.simulator.surrogate,
            pytree=model,
            replace=surrogate,
        )

        # Fit guide
        key, subkey = jr.split(key)

        (model, guide), loss_vals = fit_to_variational_target(
            key=subkey,
            dist=(model, guide),
            **guide_fit_kwargs,
        )
        losses["guide"].append(loss_vals)

    return (model, guide), losses
