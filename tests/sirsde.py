# %%

import jax.numpy as jnp
import jax.random as jr
from pyronox._temp_sirsde import SIRSDEModel, SIRSDESimulator


def test_sirsde_simulator():
    time_steps = 20
    simulator = SIRSDESimulator(time_steps=time_steps)
    params = jnp.array([0.1, 0.05, 0.05, 1000])
    y = simulator.simulate(jr.key(0), *params)
    assert y.shape == (time_steps,)
    assert simulator(jr.key(0), params).ndim == 1


def test_sirsde_model():
    key1, key2 = jr.split(jr.key(0))

    model = SIRSDEModel(key1).set_flags(reparameterized=False)
    sample1 = model.set_flags(use_surrogate=False).sample(key2)
    model.set_flags(use_surrogate=True).log_prob(sample1)
