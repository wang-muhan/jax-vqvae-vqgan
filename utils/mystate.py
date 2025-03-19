###############################
#
#  Structures for managing training of flax networks.
#
###############################

import flax
import flax.linen as nn
from flax import jax_utils, orbax_utils
import jax
import jax.numpy as jnp
from jax import tree_util
import optax
import orbax
import orbax.checkpoint
import functools
from typing import Any, Callable

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

def ema(model, target_model, tau):
    """
    Exponential moving average of model params to target_model params.
    Tau is the injection rate of the current model params into the target model params.
    """
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)

def create_checkpoint_manager(checkpoint_dir: str, max_to_keep: int = 2) -> orbax.checkpoint.CheckpointManager:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_dir, orbax_checkpointer, options)
    return checkpoint_manager

# Contains model params and optimizer state.
class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Callable = nonpytree_field()
    model: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model: nn.Module, params, tx=None, **kwargs):
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1, apply_fn=model.apply, model=model, params=params,
            tx=tx, opt_state=opt_state, **kwargs,
        )

    # Call model.apply_fn.
    def __call__(self, *args, params=None, method=None, **kwargs,):
        if params is None:
            params = self.params
        if isinstance(method, str):
            method = getattr(self.model, method)
        return self.apply_fn({"params": params}, *args, method=method, **kwargs)
    
    def __getattr__(self, name):
        if hasattr(self.model, name):
            attr = getattr(self.model, name)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    return self.model.apply({"params": self.params}, *args, method=attr, **kwargs)
                return wrapper
            else:
                return attr
        raise AttributeError(f"TrainState has no attribute {name}")

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

    def apply_loss_fn(self, *, loss_fn, pmap_axis=None, has_aux=False):
        """
        Takes a gradient step towards minimizing `loss_fn`. Internally, this calls
        `jax.grad` followed by `TrainState.apply_gradients`. If pmap_axis is provided,
        additionally it averages gradients (and info) across devices before performing update.
        """
        if has_aux:
            (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                loss = jax.lax.pmean(loss, axis_name=pmap_axis)
                info = jax.lax.pmean(info, axis_name=pmap_axis)

            return self.apply_gradients(grads=grads), loss, info

        else:
            loss, grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads), loss

    # For pickling.
    def save(self, checkpoint_dir, epoch):
        if not hasattr(self, "save_checkpoint_manager"):
            self.save_checkpoint_manager = create_checkpoint_manager(checkpoint_dir)
        save_args = orbax_utils.save_args_from_target(self)
        unreplicated_state = jax.device_get(jax.tree.map(lambda x: x[0], self))
        self.save_checkpoint_manager.save(
            epoch,
            unreplicated_state,
            save_kwargs={'save_args': save_args}
        )
        print(f"Model saved at epoch {epoch}")
    
    def load(self, checkpoint_dir, step=None):
        if not hasattr(self, "load_checkpoint_manager"):
            self.load_checkpoint_manager = create_checkpoint_manager(checkpoint_dir)
        
        # check if train from scratch
        latest_step = self.load_checkpoint_manager.latest_step() if step is None else step
        if latest_step is not None:
            # Then restore the checkpoint into this state
            state = self.load_checkpoint_manager.restore(latest_step, self)
            self.load_checkpoint_manager.wait_until_finished()
            print(f"Resuming training from epoch {latest_step}")
        else:
            print("Training from scratch")
            state = self
        return state