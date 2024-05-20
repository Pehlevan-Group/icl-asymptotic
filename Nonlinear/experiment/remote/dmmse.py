import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
import tensorflow_probability.substrates.jax.distributions as tfd

def predict(X, Y, test_x, W, scale):
    """
    Args:
        X: batch_size x i x n_dims (float)
        Y: batch_size x i x 1 (float)
        test_x: batch_size x 1 x n_dims (float)
        W: n_dims x n_tasks (float)
        scale: (float)
    Return:
        batch_size (float)
    """
    # X @ W is batch_size x i x n_tasks, Y is batch_size x i x 1, so broadcasts to alpha being batch_size x n_tasks
    alpha = tfd.Normal(0, scale).log_prob(Y - jnp.matmul(X, W, precision=lax.Precision.HIGHEST)).astype(dtype).sum(axis=1)
    # softmax is batch_size x n_tasks, W.T is n_tasks x n_dims, so w_mmse is batch_size x n_dims x 1
    w_mmse = jnp.expand_dims(jnp.matmul(jax.nn.softmax(alpha, axis=1), W.T, precision=lax.Precision.HIGHEST), -1)
    # test_x is batch_size x 1 x n_dims, so pred is batch_size x 1 x 1. NOTE: @ should be ok (batched row times column)
    pred = test_x @ w_mmse
    return pred[:, 0, 0]

def discrete_mmse(data, targets, task_pool):
    """
    Args:
        data: batch_size x n_points x n_dims (float)
        targets: batch_size x n_points (float)
    Return:
        batch_size x n_points (float)
    """
    _, n_points, _ = data.shape
    targets = jnp.expand_dims(targets, -1)  # batch_size x n_points x 1
    W = task_pool.squeeze().T  # n_dims x n_tasks  (maybe do squeeze and transpose during initialization?)
    
    preds = [data[:, 0] @ W.mean(axis=1)]  # batch_size
    preds.extend(
        [
            predict(data[:, :_i], targets[:, :_i], data[:, _i:_i+1], W, scale)
            for _i in range(1, n_points)
        ]
    )
    preds = jnp.stack(preds, axis=1)  # batch_size x n_points
    return preds


# Define global variables
d = 70; K = 5;

scale = 1.0
dtype = jnp.float32  # or any other dtype you prefer
task_pool = random.normal((d, K), dtype=dtype)

sigma = 0.1; psi = 1;
# Generate random values using numpy and convert them to jax.numpy arrays
X = jnp.array(np.sqrt(1/d) * np.random.standard_normal(size=(d, N)))
nue = jnp.array(sigma * np.random.standard_normal(size=(N, 1)))
beta = jnp.array(psi * np.random.standard_normal(size=(d, 1)))
# Compute ys
ys = X.T @ beta + nue

discrete_mmse(X, ys, task_pool)