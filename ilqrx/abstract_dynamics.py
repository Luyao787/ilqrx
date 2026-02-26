import abc
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import ilqrx.quat_utils as quat_utils

class AbstractDynamics(abc.ABC):
    """Abstract base for system dynamics."""
    def __init__(self, state_dim: int, control_dim: int, dt: float, integrator: Callable):
        """Configure the default integration settings.

        Args:
            state_dim: Dimension of the state vector.
            control_dim: Dimension of the control vector.
            dt: Discretization time step.
            integrator: Integration method, a callable.
        """
        self._nx = state_dim
        self._nu = control_dim
        self._ndx = state_dim  # tangent space dimension
        self._dt = dt
        self._integrator = integrator
        self._use_quaternion = False

    @property
    def state_dim(self) -> int:
        return self._nx

    @property
    def control_dim(self) -> int:
        return self._nu

    @property
    def tangent_space_dim(self) -> int:
        return self._ndx

    @property
    def dt(self) -> float:
        return self._dt
    
    @property
    def use_quaternion(self) -> bool:
        return self._use_quaternion

    @abc.abstractmethod
    def dynamics_ct(self, x: jnp.ndarray, u: jnp.ndarray, t: int) -> jnp.ndarray:
        """Continuous-time dynamics f(x, u, t) = dx/dt.

        Args:
            x: State vector, shape (n,).
            u: Control vector, shape (m,).
            t: Time index (can be ignored for time-invariant systems).

        Returns:
            Time derivative of the state, shape (n,).
        """
        raise NotImplementedError

    def dynamics_dt(self, x: jnp.ndarray, u: jnp.ndarray, t: int) -> jnp.ndarray:
        """Discrete-time dynamics F(x, u, t) via integration.

        Returns:
            Next state vector, shape (n,).

        """
        return self._integrator(self.dynamics_ct, dt=self._dt)(x, u, t)
    
    def dynamics_defect(self, x: jnp.ndarray, u: jnp.ndarray, t: int, x_next: jnp.ndarray) -> jnp.ndarray:
        """Dynamics defect: difference between integrated next state and given next state."""
        x_pred = self._integrator(self.dynamics_ct, dt=self._dt)(x, u, t)
        return x_pred - x_next

    def jacobian_dt(self, x: jnp.ndarray, u: jnp.ndarray, t: int, x_next: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Jacobian of the discrete-time dynamics at (x, u, t)."""
        del x_next  # unused
        F = self._integrator(self.dynamics_ct, dt=self._dt)
        A_dt = jax.jacobian(F)(x, u, t)
        B_dt = jax.jacobian(F, argnums=1)(x, u, t)
        return A_dt, B_dt
    
    def state_update(self, x: jnp.ndarray, delta_x: jnp.ndarray) -> jnp.ndarray:
        """Update state x with perturbation delta_x in the tangent space."""
        return x + delta_x
        
    def __call__(self, x: jnp.ndarray, u: jnp.ndarray, t: int) -> jnp.ndarray:
        """Discrete-time next state using the configured integrator."""
        return self._integrator(self.dynamics_ct, dt=self._dt)(x, u, t)


class DynamicsWithQuaternion(AbstractDynamics):
    """Dynamics class with quaternion utilities."""

    def __init__(self, state_dim, control_dim, quat_start_index, dt, integrator):
        super().__init__(state_dim, control_dim, dt, integrator)    
        self._quat_start_index = quat_start_index
        self._ndx = state_dim - 1
        self._use_quaternion = True

    def _compute_error_state_jacobian(self, x: jnp.ndarray) -> jnp.ndarray:
        Gq = quat_utils.attitude_jacobian(x[self._quat_start_index : self._quat_start_index + 4])
        E = jsp.linalg.block_diag(jnp.eye(self._quat_start_index), Gq, jnp.eye(self._ndx  - self._quat_start_index - 3))
        return E

    def dynamics_dt(self, x, u, t):
        x_next = super().dynamics_dt(x, u, t)
        # Normalize quaternion part
        q_next = x_next[self._quat_start_index : self._quat_start_index + 4]
        q_next_normalized = q_next / jnp.linalg.norm(q_next)
        x_next = x_next.at[self._quat_start_index : self._quat_start_index + 4].set(q_next_normalized)
        return x_next

    def dynamics_defect(self, x, u, t, x_next):
        x_pred = self.dynamics_dt(x, u, t)
        return self.difference(x_next, x_pred)
        
    def jacobian_dt(self, x, u, t, x_next):
        Ad, Bd = super().jacobian_dt(x, u, t, x_next)
        E = self._compute_error_state_jacobian(x)
        E_next = self._compute_error_state_jacobian(x_next)
        Ad_corrected = E_next.T @ Ad @ E
        Bd_corrected = E_next.T @ Bd
        return Ad_corrected, Bd_corrected
    
    def difference(self, x1, x2):
        # x2 - x1
        q1 = x1[self._quat_start_index : self._quat_start_index + 4]
        q2 = x2[self._quat_start_index : self._quat_start_index + 4]
        delta_q = quat_utils.quat_multiply(quat_utils.quat_conjugate(q1), q2)
        phi = quat_utils.inv_cayley_map(delta_q)
        delta_x = jnp.concatenate([x2[:self._quat_start_index] - x1[:self._quat_start_index], 
                                   phi, 
                                   x2[self._quat_start_index + 4:] - x1[self._quat_start_index + 4:]])
        return delta_x
    
    def state_update(self, x, delta_x):
        q = x[self._quat_start_index : self._quat_start_index + 4]
        phi = delta_x[self._quat_start_index : self._quat_start_index + 3]
        delta_q = quat_utils.cayley_map(phi)
        q_updated = quat_utils.quat_multiply(q, delta_q)
        x_updated = jnp.concatenate([x[:self._quat_start_index] + delta_x[:self._quat_start_index],
                                     q_updated,
                                     x[self._quat_start_index + 4:] + delta_x[self._quat_start_index + 3:]])
        return x_updated
    
    def __call__(self, x, u, t):
        x_next = super().dynamics_dt(x, u, t)
        # Normalize quaternion part
        q_next = x_next[self._quat_start_index : self._quat_start_index + 4]
        q_next_normalized = q_next / jnp.linalg.norm(q_next)
        x_next = x_next.at[self._quat_start_index : self._quat_start_index + 4].set(q_next_normalized)
        return x_next