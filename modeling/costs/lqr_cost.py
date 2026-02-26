import jax
import jax.numpy as jnp
import jax.scipy as jsp
from ilqrx.abstract_cost import AbstractCost
import ilqrx.quat_utils as quat_utils
from jax import tree_util

@tree_util.register_pytree_node_class
class LQRCost(AbstractCost):
    def __init__(self, Q, R, Qf, x_ref, u_ref, T):
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.T = T

    def cost_fn(self, x, u, t):
        dx = x - self.x_ref
        du = u - self.u_ref
        stage_cost = 0.5 * jnp.dot(dx, self.Q @ dx) + 0.5 * jnp.dot(du, self.R @ du)
        terminal_cost = 0.5 * jnp.dot(dx, self.Qf @ dx)
        return jnp.where(t == self.T, terminal_cost, stage_cost)
    
    def tree_flatten(self):
        children = (self.Q, self.R, self.Qf, self.x_ref, self.u_ref, self.T)
        aux_data = ()
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        Q, R, Qf, x_ref, u_ref, T = children
        return cls(Q, R, Qf, x_ref, u_ref, T)
    

    
    
@tree_util.register_pytree_node_class
class LQRCostWithQuaternion(AbstractCost):
    def __init__(self, Q, R, Qf, x_ref, u_ref, T, quat_start_index):
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.T = T
        self.quat_start_index = quat_start_index

    def cost_fn(self, x, u, t):
        dx_no_quat = jnp.concatenate([x[:self.quat_start_index] - self.x_ref[:self.quat_start_index],
                                      x[self.quat_start_index + 4:] - self.x_ref[self.quat_start_index + 4:]])
        du = u - self.u_ref
        Q_no_quat = jsp.linalg.block_diag(self.Q[:self.quat_start_index, :self.quat_start_index],
                                          self.Q[self.quat_start_index + 4:, self.quat_start_index + 4:])
        Qf_no_quat = jsp.linalg.block_diag(self.Qf[:self.quat_start_index, :self.quat_start_index],
                                           self.Qf[self.quat_start_index + 4:, self.quat_start_index + 4:])
        stage_cost = 0.5 * jnp.dot(dx_no_quat, Q_no_quat @ dx_no_quat) + 0.5 * jnp.dot(du, self.R @ du)
        terminal_cost = 0.5 * jnp.dot(dx_no_quat, Qf_no_quat @ dx_no_quat)

        # add cost related to quaternion error
        w_quat = self.Q[self.quat_start_index, self.quat_start_index]
        wf_quat = self.Qf[self.quat_start_index, self.quat_start_index]
        q = x[self.quat_start_index : self.quat_start_index + 4]
        q_ref = self.x_ref[self.quat_start_index : self.quat_start_index + 4]
        cost_quat = 1 - jnp.abs(jnp.dot(q, q_ref))
        stage_cost += w_quat * cost_quat
        terminal_cost += wf_quat * cost_quat

        return jnp.where(t == self.T, terminal_cost, stage_cost)
    
    def gradient(self, x, u, t):
        grad_x, grad_u = super().gradient(x, u, t)
        # Adjust gradient w.r.t quaternion
        q = x[self.quat_start_index : self.quat_start_index + 4]
        q_ref = self.x_ref[self.quat_start_index : self.quat_start_index + 4]
        w_quat = jnp.where(t == self.T,
                           self.Qf[self.quat_start_index, self.quat_start_index],
                           self.Q[self.quat_start_index, self.quat_start_index])
        G = quat_utils.attitude_jacobian(q)
        sign_val = jnp.where(jnp.dot(q, q_ref) >= 0, 1.0, -1.0)
        grad_quat = -w_quat * sign_val * G.T @ q_ref
        grad_x_corrected = jnp.concatenate([grad_x[:self.quat_start_index],
                                            grad_quat,
                                            grad_x[self.quat_start_index + 4:]])
        return grad_x_corrected, grad_u
    
    def hessian(self, x, u, t):
        ndx = x.shape[0] - 1
        nu = u.shape[0]
        hess_x, hess_u, _ = super().hessian(x, u, t)
        hess_xu_corrected = jnp.zeros((ndx, nu))
        # Adjust Hessian w.r.t quaternion
        q = x[self.quat_start_index : self.quat_start_index + 4]
        q_ref = self.x_ref[self.quat_start_index : self.quat_start_index + 4]
        w_quat = jnp.where(t == self.T,
                           self.Qf[self.quat_start_index, self.quat_start_index],
                           self.Q[self.quat_start_index, self.quat_start_index])
        q_dot_q_ref = jnp.dot(q, q_ref)
        sign_val = jnp.where(q_dot_q_ref >= 0, 1.0, -1.0)
        hess_quat = w_quat * sign_val * q_dot_q_ref * jnp.eye(3)

        hess_x_corrected = jnp.zeros((ndx, ndx))
        hess_x_corrected = hess_x_corrected.at[:self.quat_start_index, :self.quat_start_index].set(
            hess_x[:self.quat_start_index, :self.quat_start_index])
        hess_x_corrected = hess_x_corrected.at[self.quat_start_index : self.quat_start_index + 3,
                                                  self.quat_start_index : self.quat_start_index + 3].set(hess_quat)
        hess_x_corrected = hess_x_corrected.at[self.quat_start_index + 3:, self.quat_start_index + 3:].set(
            hess_x[self.quat_start_index + 4:, self.quat_start_index + 4:]) 

        return hess_x_corrected, hess_u, hess_xu_corrected
    
    def tree_flatten(self):
        children = (self.Q, self.R, self.Qf, self.x_ref, self.u_ref, self.T, self.quat_start_index)
        aux_data = ()
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        Q, R, Qf, x_ref, u_ref, T, quat_start_index = children
        return cls(Q, R, Qf, x_ref, u_ref, T, quat_start_index)