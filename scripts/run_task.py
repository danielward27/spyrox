import argparse
import os
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from pyrox import losses
from pyrox.program import (
    GuideToDataSpace,
)

from spyrox import utils
from spyrox.tasks.sirsde_with_covariates import get_task
from spyrox.train import rounds_based_snle


def get_losses(n_particles=8):
    """Get the loss functions under consideration."""
    return {
        "SoftCVI(a=0.75)": losses.SoftContrastiveEstimationLoss(
            n_particles=n_particles,
            alpha=0.75,
        ),
        "SoftCVI(a=1)": losses.SoftContrastiveEstimationLoss(
            n_particles=n_particles,
            alpha=1,
        ),
        "ELBO": losses.EvidenceLowerBoundLoss(n_particles=n_particles),
        "SNIS-fKL": losses.SelfNormImportanceWeightedForwardKLLoss(
            n_particles=n_particles,
        ),
    }


def run_task(
    seed: int,
    *,
    simulation_budget: int,
    num_rounds: int,
    loss_name: str,
    surrogate_fit_kwargs: dict,
    guide_fit_kwargs: dict,
):

    guide_fit_kwargs["loss_fn"] = get_losses()[loss_name]

    key, subkey = jr.split(jr.key(seed))
    model, guide = get_task(subkey)

    # model samples can be many simulations
    samples_per_round = round(simulation_budget / (num_rounds * model.program.n_obs))

    # simulated obs
    key, subkey = jr.split(key)
    true_latents = eqx.filter_jit(model.sample)(subkey, use_surrogate=False)
    obs = true_latents.pop("x")

    key, subkey = jr.split(key)

    (model, guide), losses = rounds_based_snle(
        key=subkey,
        model=model,
        guide=guide,
        num_rounds=num_rounds,
        samples_per_round=samples_per_round,  # note a model sample is >1 simulation!
        sim_param_name="z",
        surrogate_fit_kwargs=surrogate_fit_kwargs,
        guide_fit_kwargs=guide_fit_kwargs,
        obs=obs,
    )
    losses = {
        "surrogate_val": jnp.asarray(
            [loss for loss_vals in losses["surrogate"]["val"] for loss in loss_vals],
        ),
        "surrogate_train": jnp.asarray(
            [loss for loss_vals in losses["surrogate"]["train"] for loss in loss_vals],
        ),
        "guide": jnp.asarray(losses["guide"]),
    }

    key, subkey = jr.split(key)
    n_samples = 1000

    key, subkey = jr.split(key)

    joint_samples = jax.jit(jax.vmap(partial(model.sample, use_surrogate=False)))(
        jr.split(key, n_samples),
    )

    data_space_guide = GuideToDataSpace(
        guide=guide,
        model=model,
        guide_kwargs={},
        model_kwargs={"use_surrogate": False, "obs": obs},
    )

    log_prob_true = data_space_guide.log_prob(true_latents)

    guide_samps = jax.jit(jax.vmap(data_space_guide.sample))(jr.split(key, n_samples))
    guide_lps = eqx.filter_vmap(data_space_guide.log_prob)(guide_samps)

    metrics = {
        "log_prob_true": log_prob_true,
        "coverage_prob": jnp.mean(log_prob_true >= guide_lps),
    }

    key, subkey = jr.split(key)
    results_str = (
        f"{loss_name}_seed={seed}_num_rounds={num_rounds}_budget={simulation_budget}"
    )

    jnp.savez(f"./results/metrics/{results_str}.npz", **metrics)
    jnp.savez(f"./results/samples/joint_{results_str}.npz", **joint_samples)
    jnp.savez(f"./results/samples/guide_{results_str}.npz", **guide_samps)
    jnp.savez(f"./results/samples/true_{results_str}.npz", **true_latents)
    jnp.savez(f"./results/losses/{results_str}.npz", **losses)

    # Save guide
    eqx.tree_serialise_leaves(f"./results/models/{results_str}.eqx", guide)


if __name__ == "__main__":
    # Quicker to run example command (not reasonable values though!)
    # python -m scripts.run_task --seed=-1 --loss-name="SoftCVI(a=1)" --num-rounds=2 --simulation-budget=100 --guide-steps=10 --surrogate-max-epochs=10 --show-progress

    # Note guide steps is total.

    os.chdir(utils.get_abspath_project_root())
    parser = argparse.ArgumentParser(description="softcvi")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--loss-name", type=str, required=True)
    parser.add_argument("--num-rounds", type=int, required=True)
    parser.add_argument("--simulation-budget", type=int, required=True)
    parser.add_argument("--guide-steps", type=int, required=True)
    parser.add_argument("--surrogate-max-epochs", type=int, required=True)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    surrogate_fit_kwargs = {
        "optimizer": optax.apply_if_finite(
            optax.chain(
                optax.adam(2e-4),
                optax.clip_by_global_norm(10),
            ),
            max_consecutive_errors=10,
        ),
        "max_patience": 5,
        "batch_size": 50,
        "show_progress": args.show_progress,
        "max_epochs": 300,
    }

    guide_fit_kwargs = {
        "optimizer": optax.apply_if_finite(
            optax.chain(
                optax.adam(3e-4),
                optax.clip_by_global_norm(10),
            ),
            max_consecutive_errors=10,
        ),
        "steps": args.guide_steps // args.num_rounds,
        "show_progress": args.show_progress,
    }

    run_task(
        seed=args.seed,
        simulation_budget=args.simulation_budget,
        num_rounds=args.num_rounds,
        loss_name=args.loss_name,
        surrogate_fit_kwargs=surrogate_fit_kwargs,
        guide_fit_kwargs=guide_fit_kwargs,
    )
