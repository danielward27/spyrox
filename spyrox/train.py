from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flowjax import wrappers
from flowjax.distributions import AbstractDistribution

# TODO consider just using flowjax.fit_variational
from flowjax.train import fit_to_key_based_loss
from flowjax.train.loops import fit_to_data, fit_to_key_based_loss
from jaxtyping import Array, PRNGKeyArray, PyTree
from pyrox.program import AbstractProgram, SetKwargs

from spyrox.simulator import AbstactProgramWithSurrogate, SimulatorToDistribution


class ProgramToProgramSamplesLoss(eqx.Module):
    """Fits a program to samples from another program using maximum likelihood."""

    program_to_sample: AbstractProgram

    def __init__(self, program_to_sample: AbstractProgram):
        self.program_to_sample = program_to_sample

    def __call__(self, params: PyTree, static: PyTree, key: PRNGKeyArray):
        program = wrappers.unwrap(eqx.combine(params, static))
        sample = self.program_to_sample.sample(key)
        loss = -program.log_prob(sample)
        eqx.debug.breakpoint_if(~jnp.isfinite(loss))  # TODO
        return loss


def get_surrogate(program: AbstractProgram) -> AbstractDistribution:
    "Maps across program, in case wrapper class used."
    leaves = jax.tree.leaves(
        program,
        is_leaf=lambda leaf: isinstance(leaf, AbstactProgramWithSurrogate),
    )
    surrogates = [
        leaf.surrogate
        for leaf in leaves
        if isinstance(leaf, AbstactProgramWithSurrogate)
    ]
    if len(surrogates) == 0:
        raise ValueError("No SimulatorToDistribution found in program.")
    if len(surrogates) > 1:
        raise ValueError("Multiple simulators not supported.")
    return surrogates[0]


def set_surrogate(
    program: AbstactProgramWithSurrogate,
    surrogate: AbstractDistribution,
):
    def replace_fn(leaf):
        if isinstance(leaf, AbstactProgramWithSurrogate):
            return eqx.tree_at(lambda p: p.surrogate, leaf, surrogate)
        return leaf

    return jax.tree.map(
        replace_fn,
        program,
        is_leaf=lambda leaf: isinstance(leaf, AbstactProgramWithSurrogate),
    )


def rounds_based_snle(
    key: PRNGKeyArray,
    model: AbstractProgram,
    guide: AbstractProgram,
    num_rounds: int,
    sim_per_round: int,
    surrogate_fit_kwargs: dict,
    guide_fit_kwargs: dict,
    simulator_param_name: str,
):
    # We assume the model has a use_surrogate.
    # We assume model has attribute model.simulator
    # We assume the model has a vector (e.g. deterministic site)?
    # We assume the last dimensions of simulations are the shape of the simulator
    # not e.g. a plate

    losses = {
        "surrogate": {"train": [], "val": []},
        "guide": [],
    }

    def _vmap(fn, **kwargs):  # filter_vmap with kwargs
        return eqx.filter_vmap(partial(fn))

    # infer obs name
    obs_name = model.site_names(use_surrogate=False).observed
    assert len(obs_name) == 1
    obs_name = list(obs_name)[0]

    @eqx.filter_jit
    def simulate(
        key,
        proposal,
        model,
    ):

        def simulate_single(key):
            proposal_key, sim_key = jr.split(key)
            sim_params = proposal.sample(proposal_key)[simulator_param_name]
            simulation = model.sample(sim_key, use_surrogate=False)[obs_name]
            return sim_params, simulation

        key, subkey = jr.split(key)
        sim_params, simulations = jax.vmap(simulate_single)(
            jr.split(subkey, sim_per_round),
        )

        # Reshape to remove any plates
        simulations = simulations.reshape(-1, *get_surrogate(model).shape)
        sim_params = sim_params.reshape(-1, *get_surrogate(model).cond_shape)
        return simulations, sim_params

    for _ in range(num_rounds):
        simulations, sim_params = simulate(key, guide, model)

        if not jnp.isfinite(simulations).all():
            raise ValueError("Inf or nan values detected in simulations.")

        # Fit simulator likelihood
        key, subkey = jr.split(key)
        surrogate, loss_vals = fit_to_data(
            subkey,
            dist=get_surrogate(model),
            x=simulations,
            condition=sim_params,
            **surrogate_fit_kwargs,
        )
        losses["surrogate"]["train"].append(loss_vals["train"])
        losses["surrogate"]["val"].append(loss_vals["val"])

        model = set_surrogate(model, surrogate)

        # Fit guide
        key, subkey = jr.split(key)
        model = SetKwargs(model, use_surrogate=True)

        (model, guide), loss_vals = fit_to_key_based_loss(
            key=subkey,
            tree=(wrappers.non_trainable(model), guide),
            **guide_fit_kwargs,
        )
        model = model.program  # Undo set kwargs
        losses["guide"].append(loss_vals)

    return (model, guide), losses


# TODO simulator should probably be seperate to model?
# def continuous_training(
#     *,
#     key: PRNGKeyArray,
#     model: AbstractProgram,
#     guide: AbstractProgram,
#     steps: int,
#     obs: Array,
#     surrogate_optimizer: optax.GradientTransformation,
#     vi_optimizer: optax.GradientTransformation,
#     simulator_param_name: str,
#     obs_name: str = "obs",
#     show_progress: bool = True,
# ):
#     losses = {
#         "surrogate": [],
#         "vi": [],
#     }

#     params, static = eqx.partition(
#         (model, guide),
#         eqx.is_inexact_array,
#         is_leaf=lambda leaf: isinstance(leaf, wrappers.NonTrainable),
#     )

#     surrogate_loss_fn = partial(
#         fjlosses.MaximumLikelihoodLoss(),
#         static=static[0].simulator.surrogate,
#     )

#     vi_loss_fn = partial(
#         SoftContrastiveEstimationLoss(n_particles=2, alpha=0.75),
#         obs=obs,
#         static=static,
#     )

#     opt_states = {
#         "surrogate": surrogate_optimizer.init(params[0].simulator.surrogate),
#         "vi": vi_optimizer.init(params),
#     }

#     simulator_in_size = model.simulator.cond_shape[0]
#     simulator_out_size = model.simulator.shape[0]

#     @eqx.filter_jit
#     def sample_proposal_joint(key, params):
#         model, guide = eqx.combine(params, static)

#         key, subkey = jr.split(key)
#         latents = guide.sample(subkey, obs=obs)

#         key, subkey = jr.split(key)
#         simulations = model.sample(subkey, condition=latents)[obs_name]
#         sim_params = latents[simulator_param_name]

#         # Simulations and parameters may have shape (n_sim, plate, out_dim) so reshape
#         simulations = simulations.reshape(-1, simulator_out_size)
#         sim_params = sim_params.reshape(-1, simulator_in_size)
#         return simulations, sim_params

#     @eqx.filter_jit
#     def combined_steps(
#         params,
#         *,
#         key,
#         opt_states,  # dict wth (surrogate, vi)
#         sim_params,
#         simulation,
#     ):

#         surrogate_params = params[0].simulator.surrogate
#         losses = {}

#         # Surrogate step
#         surrogate_params, opt_states["surrogate"], losses["surrogate"] = step(
#             surrogate_params,
#             x=simulation,
#             condition=sim_params,
#             optimizer=surrogate_optimizer,
#             opt_state=opt_states["surrogate"],
#             loss_fn=surrogate_loss_fn,
#         )

#         # Update model with new surrogate
#         params = eqx.tree_at(
#             where=lambda p: p[0].simulator.surrogate,
#             pytree=params,
#             replace=surrogate_params,
#         )

#         # TODO ensure surrogate is not trainable in VI objective
#         # (model may have trainable parameters in general though)

#         # VI step
#         params, opt_states["vi"], losses["vi"] = step(
#             params,
#             key=key,
#             optimizer=vi_optimizer,
#             opt_state=opt_states["vi"],
#             loss_fn=vi_loss_fn,
#         )

#         return params, opt_states, losses

#     for _ in trange(steps, disable=not show_progress):

#         total_start = time()

#         start_time = time()
#         simulations, sim_params = jax.block_until_ready(
#             sample_proposal_joint(key, params),
#         )
#         print("simulations:", time() - start_time)

#         if not jnp.isfinite(simulations).all():
#             print("Warning: simulation had infs, skipping optimization step.")
#             continue

#         start_time = time()

#         params, opt_states, losses_i = jax.block_until_ready(
#             combined_steps(
#                 params,
#                 key=key,
#                 opt_states=opt_states,
#                 sim_params=sim_params,
#                 simulation=simulations,
#             )
#         )
#         print("combined_steps:", time() - start_time)

#         start_time = time()
#         losses["surrogate"].append(losses_i["surrogate"])
#         losses["vi"].append(losses_i["vi"])
#         print("appending_to_losses:", time() - start_time)

#         print("total", time() - total_start)

#     return eqx.combine(params, static), losses
