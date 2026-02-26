import jax.numpy as jnp
import ilqrx.quat_utils as quat_utils
from ilqrx.abstract_dynamics import DynamicsWithQuaternion

class Quadrotor(DynamicsWithQuaternion):
    """Quadrotor dynamics in continuous time."""
    def __init__(self, dt, integrator):
        super().__init__(state_dim=13, control_dim=4, quat_start_index=3, integrator=integrator, dt=dt)
        self.mass = 1.0
        self.J = jnp.diag(jnp.array([0.01, 0.01, 0.02]))
        self.J_inv = jnp.linalg.inv(self.J)

    def dynamics_ct(self, x, u, t):
        del t
        # pos = x[0:3]
        quat = x[3:7]
        lin_vel = x[7:10]
        ang_vel = x[10:13]
        
        f = u[0]
        tau = u[1:4]

        pos_dot = lin_vel

        L = quat_utils.left_matrix(quat)
        ang_vel_hat = jnp.zeros(4)
        ang_vel_hat = ang_vel_hat.at[1:4].set(ang_vel)
        quat_dot = 0.5 * L @ ang_vel_hat
        
        gravity = jnp.array([0., 0., -9.81])
        R = quat_utils.rotation_matrix_from_quat(quat)
        lin_vel_dot = (R @ jnp.array([0., 0., f])) / self.mass + gravity
        ang_vel_dot = self.J_inv @ (tau - jnp.cross(ang_vel, self.J @ ang_vel))

        x_dot = jnp.concatenate([pos_dot, quat_dot, lin_vel_dot, ang_vel_dot])

        return x_dot