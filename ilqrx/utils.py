import jax
import jax.numpy as jnp
from jax import jacobian, hessian
from jax import vmap
from jax import lax
from functools import partial

pad = lambda A: jnp.vstack((A, jnp.zeros((1,) + A.shape[1:])))

def vectorize(fun, argnums=3):
  """Returns a jitted and vectorized version of the input function.

  See https://jax.readthedocs.io/en/latest/jax.html#jax.vmap

  Args:
    fun: a numpy function f(*args) to be mapped over.
    argnums: number of leading arguments of fun to vectorize.

  Returns:
    Vectorized/Batched function with arguments corresponding to fun, but extra
    batch dimension in axis 0 for first argnums arguments (x, u, t typically).
    Remaining arguments are not batched.
  """
  
  def vfun(*args):
    _fun = lambda tup, *margs: fun(*(margs + tup))    
    return vmap(
        _fun, in_axes=(None,) + (0,) * argnums)(args[argnums:], *args[:argnums])

  return vfun

def linearize(fun, argnums=3):
  """Vectorized gradient or jacobian operator.

  Args:
    fun: numpy scalar or vector function with signature fun(x, u, t, *args).
    argnums: number of leading arguments of fun to vectorize.

  Returns:
    A function that evaluates Gradients or Jacobians with respect to states and
    controls along a trajectory, e.g.,

        dynamics_jacobians = linearize(dynamics)
        cost_gradients = linearize(cost)
        A, B = dynamics_jacobians(X, pad(U), timesteps)
        q, r = cost_gradients(X, pad(U), timesteps)

        where,
          X is [T+1, n] state trajectory,
          U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
          timesteps is typically np.arange(T+1)

          and A, B are Dynamics Jacobians wrt state (x) and control (u) of
          shape [T+1, n, n] and [T+1, n, m] respectively;

          and q, r are Cost Gradients wrt state (x) and control (u) of
          shape [T+1, n] and [T+1, m] respectively.

          Note: due to padding of U, last row of A, B, and r may be discarded.
  """
  jacobian_x = jacobian(fun)
  jacobian_u = jacobian(fun, argnums=1)

  def linearizer(*args):
    return jacobian_x(*args), jacobian_u(*args)

  return vectorize(linearizer, argnums)

def quadratize(fun, argnums=3):
  """Vectorized Hessian operator for a scalar function.

  Args:
    fun: numpy scalar with signature fun(x, u, t, *args).
    argnums: number of leading arguments of fun to vectorize.

  Returns:
    A function that evaluates Hessians with respect to state and controls along
    a trajectory, e.g.,

      Q, R, M = quadratize(cost)(X, pad(U), timesteps)

     where,
          X is [T+1, n] state trajectory,
          U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
          timesteps is typically np.arange(T+1)

    and,
          Q is [T+1, n, n] Hessian wrt state: partial^2 fun/ partial^2 x,
          R is [T+1, m, m] Hessian wrt control: partial^2 fun/ partial^2 u,
          M is [T+1, n, m] mixed derivatives: partial^2 fun/partial_x partial_u
  """
  hessian_x = hessian(fun)
  hessian_u = hessian(fun, argnums=1)
  hessian_x_u = jacobian(jax.grad(fun), argnums=1)

  def quadratizer(*args):
    return hessian_x(*args), hessian_u(*args), hessian_x_u(*args)

  return vectorize(quadratizer, argnums)

@partial(jax.jit, static_argnums=(0,))
def rollout(dynamics, U, x0):
  """Rolls-out x[t+1] = dynamics(x[t], U[t], t), x[0] = x0.

  Args:
    dynamics: a function f(x, u, t) to rollout.
    U: (T, m) np array for control sequence.
    x0: (n, ) np array for initial state.

  Returns:
     X: (T+1, n) state trajectory.
  """
  
  return _rollout(dynamics, U, x0)

def _rollout(dynamics, U, x0, *args):
  def dynamics_for_scan(x, ut):
    u, t = ut
    x_next = dynamics(x, u, t, *args)
    return x_next, x_next

  return jnp.vstack(
      (x0, lax.scan(dynamics_for_scan, x0, (U, jnp.arange(U.shape[0])))[1]))

def evaluate(cost, X, U, *args):
    """Evaluates cost(x, u, t) along a trajectory.

    Args:
        cost: cost_fn with signature cost(x, u, t, *args)
        X: (T, n) state trajectory.
        U: (T, m) control sequence.
        *args: args for cost_fn

    Returns:
        objectives: (T, ) array of objectives.
    """
    timesteps = jnp.arange(X.shape[0])  
    return vectorize(cost, argnums=3)(X, U, timesteps, *args)

@jax.jit
def project_psd_cone(Q, delta=0.0):
  """Projects to the cone of positive semi-definite matrices.

  Args:
    Q: [n, n] symmetric matrix.
    delta: minimum eigenvalue of the projection.

  Returns:
    [n, n] symmetric matrix projection of the input.
  """
  S, V = jnp.linalg.eigh(Q)
  S = jnp.maximum(S, delta)
  Q_plus = jnp.matmul(V, jnp.matmul(jnp.diag(S), V.T))
  return 0.5 * (Q_plus + Q_plus.T)


@partial(jax.jit, static_argnums=(0,))
def traj_cost(cost_fn, X, U):
    evaluator = partial(evaluate, cost_fn)
    costs = evaluator(X, pad(U))
    return jnp.sum(costs)


def compute_expected_cost_reduction(delta_X, delta_U, Qs, Rs, Ms):
    def stage_quad_cost(dx, du, Q, R, M):
        quad_cost = 0.5 * (dx.T @ Q @ dx + du.T @ R @ du) + du.T @ M @ dx
        return quad_cost
    stage_quad_costs = jax.vmap(stage_quad_cost)(delta_X[:-1], delta_U, Qs[:-1], Rs, Ms)
    terminal_quad_cost = 0.5 * delta_X[-1].T @ Qs[-1] @ delta_X[-1]
    return jnp.sum(stage_quad_costs) + terminal_quad_cost


def regularize(Q, R, M, psd_delta):
    T = Q.shape[0] - 1
    n = Q.shape[1]
    psd = jax.vmap(partial(project_psd_cone, delta=psd_delta))
    # This is done to ensure that the R are positive definite.
    R = psd(R)
    # This is done to ensure that the Q - M R^(-1) M^T are positive semi-definite.
    Rinv = jax.vmap(lambda t: jnp.linalg.inv(R[t]))(jnp.arange(T))
    MRinvMT = jax.vmap(lambda t: M[t] @ Rinv[t] @ M[t].T)(jnp.arange(T))
    QMRinvMT = jax.vmap(lambda t: Q[t] - MRinvMT[t])(jnp.arange(T))
    QMRinvMT = psd(QMRinvMT)
    Q_T = Q[T].reshape([1, n, n])
    Q_T = psd(Q_T)
    Q = jnp.concatenate([QMRinvMT + MRinvMT, Q_T])
    return Q, R


def lagrangian(cost, dynamics, x0):
    """Returns a function to evaluate Lagrangian.

    L(X, U, Y) = y_0^T (\hat{x}_0 - x_0) + \sum_{k=0}^{N-1} [ l(x_k, u_k) + y_{k+1}^T ( f(x_k, u_k) - x_{k+1} ) ] + l(x_N)
    l(x_k, u_k) + y_{k+1}^T f(x_k, u_k) + y_k^T (\hat{x}_0 - x_k)   k == 0
    l(x_k, u_k) + y_{k+1}^T f(x_k, u_k) - y_k^T x_k                 k == 1,...,N-1
    l(x_N)                                                          k == N
    
    """
    def fun(x, u, t, y, y_next):
        c1 = cost(x, u, t)
        c2 = jnp.dot(y_next, dynamics(x, u, t))
        c3 = jnp.dot(y, lax.select(t == 0, x0 - x, -x))
        return c1 + c2 + c3
    return fun


def compute_defects(dynamics_defect, X, U):   
    timesteps = jnp.arange(X.shape[0] - 1) 
    return vectorize(dynamics_defect, argnums=4)(X[:-1], U, timesteps, X[1:])

def compute_dynamics_jacobians(jacobian, X, U):
    timesteps = jnp.arange(U.shape[0])
    return jax.vmap(jacobian, in_axes=(0, 0, 0, 0))(X[:-1], U, timesteps, X[1:])

def compute_norm(vec, order=1):
    return jnp.linalg.norm(vec.ravel(), ord=order) 