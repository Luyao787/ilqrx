import jax.numpy as jnp
from ilqrx.abstract_dynamics import AbstractDynamics

class QuadPend(AbstractDynamics):
    """Quadruple pendulum dynamics in continuous time."""
    def __init__(self, dt, integrator):
        super().__init__(state_dim=8, control_dim=2, integrator=integrator, dt=dt)
        self.Mass = 0.486
        self.mass = 0.2 * self.Mass
        self.grav = 9.81
        self.l = 0.25
        self.L = 2 * self.l
        self.J = 0.00383
        self.fric = 0.01

    def dynamics_ct(self, x, u, t):
        del t
        q = x[:4]
        q_dot = x[4:]
        phi = q[3]

        M_q = jnp.zeros((4, 4))
        M_q = M_q.at[0, 0].set(self.Mass + self.mass)
        M_q = M_q.at[1, 1].set(self.Mass + self.mass)
        M_q = M_q.at[2, 2].set(self.J)
        M_q = M_q.at[3, 3].set(self.mass * self.L * self.L)
        M_q = M_q.at[0, 3].set(self.mass * self.L * jnp.cos(phi))
        M_q = M_q.at[1, 3].set(self.mass * self.L * jnp.sin(phi))
        M_q = M_q.at[3, 0].set(self.mass * self.L * jnp.cos(phi))
        M_q = M_q.at[3, 1].set(self.mass * self.L * jnp.sin(phi))

        M_inv = jnp.linalg.inv(M_q)

        # Generalized forces
        torque_fric_pole = -self.fric * (q_dot[3] - q_dot[2])
        F_q = jnp.hstack([
            -jnp.sum(u) * jnp.sin(q[2]),
            jnp.sum(u) * jnp.cos(q[2]),
            (u[0] - u[1]) * self.l - torque_fric_pole,
            torque_fric_pole
        ])

        # Alternative approach: Direct computation of Coriolis and centrifugal forces
        # Compute M_dot * q_dot (Coriolis/centrifugal terms)
        phi_dot = q_dot[3]
        
        q_ddot = jnp.zeros(4)
        q_ddot = q_ddot.at[0].set(self.mass * self.L * phi_dot * phi_dot * jnp.sin(phi))
        q_ddot = q_ddot.at[1].set(-(self.Mass + self.mass) * self.grav - self.mass * self.L * phi_dot * phi_dot * jnp.cos(phi))
        q_ddot = q_ddot.at[2].set(0)
        q_ddot = q_ddot.at[3].set(-self.mass * self.L * self.grav * jnp.sin(phi))
        q_ddot += F_q
        q_ddot = M_inv @ q_ddot

        x_dot = jnp.hstack([q_dot, q_ddot]) 
        
        return x_dot