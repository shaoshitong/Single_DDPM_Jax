# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import flax
from jax import lax
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax.training import train_state
from models import utils as mutils
from diffusions.ddpm import DDPM

def batch_add(a, b):
  return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)


def schedule_fn(lr, step, config):
    warmup = config.optim.warmup
    if warmup > 0:
        lr = lr * jnp.minimum(step / warmup, 1.0)
    return lr


def get_optimizer(config):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optax.chain(optax.scale_by_schedule(lambda step: schedule_fn(lr=config.optim.lr, config=config, step=step)),
                                optax.adam(learning_rate=config.optim.lr, b1=config.optim.beta1, eps=config.optim.eps))
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(state,
                    grad,
                    new_model_state,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        lr = state.lr
        # if warmup > 0:
        #     lr = lr * jnp.minimum(state.step / warmup, 1.0)
        # if grad_clip >= 0:
        #     # Compute global gradient norm
        #     grad_norm = jnp.sqrt(
        #         sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grad)]))
        #     # Clip gradient
        #     clipped_grad = jax.tree_map(
        #         lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
        # else:  # disabling gradient clipping if grad_clip < 0
        #     clipped_grad = grad
        return state.optimizer.apply_gradients(grads=grad)

    return optimize_fn





def get_ddpm_loss_fn(ddpm, model, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(ddpm, DDPM), "DDPM training only works for DDPM."

    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch['image']
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, ddpm.N, shape=(data.shape[0],))
        sqrt_alphas_cumprod = ddpm.sqrt_alphas_cumprod
        sqrt_betas_cumprod = ddpm.sqrt_betas_cumprod
        rng, step_rng = random.split(rng)
        noise = random.normal(step_rng, data.shape)
        perturbed_data = batch_mul(sqrt_alphas_cumprod[labels], data) + \
                         batch_mul(sqrt_betas_cumprod[labels], noise)
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        losses = jnp.square(score - noise)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn



def get_step_fn(sde, model, train, optimize_fn=None, reduce_mean=False):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training and `False` for evaluation.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """
    if isinstance(sde,DDPM):
        loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    else:
        raise NotImplementedError

    def step_fn(carry_state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """

        (rng, state) = carry_state
        rng, step_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
        if train:

            def log():
                param_norm = jnp.sqrt(
                    sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(state.optimizer.params)]))
                jax.debug.print("param_norm 1 {x}", x = param_norm)
            def no_log():
                pass
            lax.cond(jnp.equal(jnp.mod(jnp.array(state.step),100),jnp.array(0)),log,no_log)

            params = state.optimizer.params
            states = state.model_state
            (loss, new_model_state), grad = grad_fn(step_rng, state.optimizer.params, states, batch)
            grad = jax.lax.pmean(grad, axis_name='batch')
            new_optimizer = optimize_fn(state, grad, new_model_state)

            def log():
                param_norm = jnp.sqrt(
                    sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(new_optimizer.params)]))
                jax.debug.print("param_norm 2 {x}", x = param_norm)
            def no_log():
                pass
            lax.cond(jnp.equal(jnp.mod(jnp.array(state.step),100),jnp.array(0)),log,no_log)

            new_params_ema = jax.tree_map(
                lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
                state.params_ema, new_optimizer.params
            )
            step = state.step + 1
            new_state = state.replace(
                step=step,
                optimizer=new_optimizer,
                model_state=new_model_state,
                params_ema=new_params_ema
            )
        else:
            loss, _ = loss_fn(step_rng, state.params_ema, state.model_state, batch)
            new_state = state

        loss = jax.lax.pmean(loss, axis_name='batch')
        new_carry_state = (rng, new_state)
        def log():
            param_norm = jnp.sqrt(
                sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(new_state.optimizer.params)]))
            jax.debug.print("param_norm 3 {x}", x = param_norm)
        def no_log():
            pass
        lax.cond(jnp.equal(jnp.mod(jnp.array(state.step),100),jnp.array(0)),log,no_log)
        return new_carry_state, loss

    return step_fn
