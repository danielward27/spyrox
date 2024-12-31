import equinox as eqx
import jax
import jax.random as jr
import paramax
from flowjax.distributions import AbstractDistribution

# TODO consider just using flowjax.fit_variational
from flowjax.train import fit_to_key_based_loss
from flowjax.train.loops import fit_to_data, fit_to_key_based_loss
from jaxtyping import Array, PRNGKeyArray, PyTree
from pyrox.program import AbstractProgram, SetKwargs

from spyrox.simulator import AbstactProgramWithSurrogate


class ProgramToProgramSamplesLoss(eqx.Module):
    """Fits a program to samples from another program using maximum likelihood."""

    program_to_sample: AbstractProgram

    def __init__(self, program_to_sample: AbstractProgram):
        self.program_to_sample = program_to_sample

    def __call__(self, params: PyTree, static: PyTree, key: PRNGKeyArray):
        program = paramax.unwrap(eqx.combine(params, static))
        sample = self.program_to_sample.sample(key)
        return -program.log_prob(sample)


def get_surrogate(program: AbstractProgram) -> AbstractDistribution:
    """Maps across program, in case wrapper class used."""
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
    samples_per_round: int,
    sim_param_name: str,
    surrogate_fit_kwargs: dict,
    guide_fit_kwargs: dict,
    obs: Array,
):
    # We assume the model includes an AbstactProgramWithSurrogate.
    # We assume model has attribute model.simulator
    # We assume the model has a site containing a vector of simulation parameters.
    # We assume the last dimensions of simulations are the shape of the simulator
    # not e.g. a plate
    # We assume the guide and model accept obs as key word arguments.

    guide = SetKwargs(guide, obs=obs)

    losses = {
        "surrogate": {"train": [], "val": []},
        "guide": [],
    }

    # infer obs name
    obs_name = model.site_names(use_surrogate=False, obs=obs).observed
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
            proposal_sample = proposal.sample(proposal_key)
            simulation = model.sample(
                sim_key,
                use_surrogate=False,
                condition=proposal_sample,
            )
            return simulation[sim_param_name], simulation[obs_name]

        key, subkey = jr.split(key)
        sim_params, simulations = jax.vmap(simulate_single)(
            jr.split(subkey, samples_per_round),
        )
        # Reshape to remove any plates
        simulations = simulations.reshape(-1, *get_surrogate(model).shape)
        sim_params = sim_params.reshape(-1, *get_surrogate(model).cond_shape)
        return simulations, sim_params

    for i in range(num_rounds):

        if i == 0:
            proposal = model.get_prior(observed_sites=(obs_name,))
            proposal = SetKwargs(proposal, use_surrogate=True)  # Unused
        else:
            proposal = guide

        key, subkey = jr.split(key)
        simulations, sim_params = simulate(key, proposal, model)

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

        (_, guide), loss_vals = fit_to_key_based_loss(
            key=subkey,
            tree=(
                paramax.non_trainable(SetKwargs(model, use_surrogate=True, obs=obs)),
                guide,
            ),
            **guide_fit_kwargs,
        )

        losses["guide"].append(loss_vals)

    # Unwrap the guide from SetKwargs wrapper
    guide = guide.program
    return (model, guide), losses
