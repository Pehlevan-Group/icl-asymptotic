"""
Model definitions
"""

# <codecell>
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import train_state

def new_seed(): return np.random.randint(1, np.iinfo(np.int32).max)

@struct.dataclass
class Metrics:
    accuracy: float
    loss: float
    l1_loss: float
    count: int = 0

    @staticmethod
    def empty():
        return Metrics(accuracy=-1, loss=-1, l1_loss=-1)
    
    def merge(self, other):
        total = self.count + 1
        acc = (self.count / total) * self.accuracy + (1 / total) * other.accuracy
        loss = (self.count / total) * self.loss + (1 / total) * other.loss
        l1_loss = (self.count / total) * self.l1_loss + (1 / total) * other.l1_loss
        return Metrics(acc, loss, l1_loss, count=total)


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(rng, model, dummy_input, lr=1e-4, optim=optax.adamw, **opt_kwargs):
    devices = jax.devices()
    if any(device.device_kind == 'gpu' for device in devices):
        print("create_train_state is running on a GPU.")
    else:
        print("create_train_state is not running on a GPU.")
    params = model.init(rng, dummy_input)['params']
    #print("creattrainstate: model init works")
    tx = optim(learning_rate=lr, **opt_kwargs)
    #print("creattrainstate: optaxadam works")

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty()
    )

def parse_loss_name(loss):
    loss_func = None
    if loss == 'bce':
        loss_func = optax.sigmoid_binary_cross_entropy
    elif loss == 'ce':
        loss_func = optax.softmax_cross_entropy_with_integer_labels
    elif loss == 'mse':
        loss_func = optax.squared_error #optax.l2_loss gives half mse error
    else:
        raise ValueError(f'unrecognized loss name: {loss}')
    return loss_func

# TODO: more robustly signal need for L1 loss
def l1_loss(params):
    # sum_params = jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), jax.tree_util.tree_leaves(params))
    # return jnp.sum(jnp.array(sum_params))
    loss = 0
    for name in params:
        if 'MBlock' in name:
            z_weights = params[name]['DenseMultiply']['kernel']
            loss += jnp.sum(jnp.abs(z_weights))

    return loss

def l2_loss(params):
    # sum_params = jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), jax.tree_util.tree_leaves(params))
    # return jnp.sum(jnp.array(sum_params))
    loss = 0
    for name in params:
        if 'MBlock' in name:
            z_weights = params[name]['DenseMultiply']['kernel']
            loss += jnp.sum(jnp.square(z_weights))

    return loss


@partial(jax.jit, static_argnames=('loss',))
def train_step(state, batch, loss='bce', l1_weight=0, l2_weight=0):
    loss_name = loss
    x, labels = batch
    loss_func = parse_loss_name(loss_name)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = loss_func(logits, labels)

        if loss_name == 'bce' and len(labels.shape) > 1:
            assert logits.shape == loss.shape
            loss = loss.mean(axis=-1)

        assert len(loss.shape) == 1

        l1_term = l1_weight * l1_loss(params)
        l2_term = l2_weight * l2_loss(params)
        return loss.mean() + l1_term + l2_term
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@partial(jax.jit, static_argnames=('loss',))
def compute_metrics(state, batch, loss='bce'):
    x, labels = batch
    logits = state.apply_fn({'params': state.params}, x)
    loss_func=parse_loss_name(loss)
    loss = loss_func(logits, labels).mean()
    l1 = l1_loss(state.params)

    if len(logits.shape) == 1:
        preds = logits > 0
    else:
        preds = logits.argmax(axis=1)
    
    if len(labels.shape) > 1:
        labels = labels.argmax(axis=1)
    
    acc = jnp.mean(preds == labels)

    metrics = Metrics(accuracy=acc, loss=loss, l1_loss=l1)
    metrics = state.metrics.merge(metrics)
    state = state.replace(metrics=metrics)
    return state

def get_random_batch(data, batch_size):
    xs,ys = data
    indices = np.random.choice(len(xs), batch_size, replace=False)
    return xs[indices],ys[indices]

def train(config, data_iter, batch_size, 
          test_iter=None, 
          loss='ce', 
          train_iters=10_000, test_iters=100, test_every=1_000, 
          early_stop_n=None, early_stop_key='loss', early_stop_decision='min',
          optim=optax.adamw,
          seed=None, 
          l1_weight=0, l2_weight=0, **opt_kwargs):

    if seed is None:
        seed = new_seed()
    
    if test_iter is None:
        test_iter = data_iter
    
    init_rng = jax.random.key(seed)
    model = config.to_model()

    samp = next(data_iter)
    mini_samp_x, _ = get_random_batch(samp, batch_size)
    state = create_train_state(init_rng, model, mini_samp_x, optim=optim, **opt_kwargs) # the samp_x data is not used during this initialisation

    hist = {
        'train': [],
        'test': []
    }

    # sample from data class, this sample will be the only one used during training
    # this is in contrast to the online training implemented in Will's train.py
    mybatch = next(data_iter)

    for step in range(train_iters):
        for _ in range(1 + len(mybatch[-1]) // batch_size):
            minibatchi = get_random_batch(mybatch, batch_size)
            state = train_step(state, minibatchi, loss=loss, l1_weight=l1_weight, l2_weight=l2_weight)
        state = compute_metrics(state, mybatch, loss=loss) 

        if (step + 1) % test_every == 0:
            hist['train'].append(state.metrics)
            state = state.replace(metrics=Metrics.empty()) 
            
            test_state = state
            for _, test_batch in zip(range(test_iters), test_iter):
                test_state = compute_metrics(test_state, test_batch, loss=loss)
            
            hist['test'].append(test_state.metrics)
            
            _print_status(step+1, hist)
            if early_stop_n is not None and len(hist['train']) > early_stop_n:
                last_n_metrics = np.array([getattr(m, early_stop_key) for m in hist['train'][-early_stop_n - 1:]])
                if early_stop_decision == 'min' and np.all(last_n_metrics[0] < last_n_metrics[1:]) \
                or early_stop_decision == 'max' and np.all(last_n_metrics[0] > last_n_metrics[1:]):
                    print(f'info: stopping early with {early_stop_key} =', last_n_metrics[-1])
                    break
    
    return state, hist

            
def _print_status(step, hist):
    print(f'ITER {step}:  loss={hist["test"][-1].loss:.4f}   l1_loss={hist["test"][-1].l1_loss:.4f}  acc={hist["test"][-1].accuracy:.4f}')

