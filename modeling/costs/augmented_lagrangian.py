import jax
import jax.numpy as jnp
from jax import tree_util
from ilqrx.abstract_cost import AbstractCost

# def _compute_active_set(dual_var: jnp.ndarray, constraint_val: jnp.ndarray) -> jnp.ndarray:
#     return jnp.invert(jnp.isclose(dual_var, 0.0) & (constraint_val < 0.0))

# class AugmentedLagrangian(AbstractCost):
#     def __init__(self, base_cost_model, constraint_model, Y_eq, Y_ineq, rho):
#         self._base_cost_model = base_cost_model
#         self._constraint_model = constraint_model
#         self._Y_eq = Y_eq
#         self._Y_ineq = Y_ineq
#         self._rho = rho

#     def cost_fn(self, x, u, t):
#         cost = self._base_cost_model.cost_fn(x, u, t)
#         ineq = self._constraint_model.constraint_fn(x, u, t)
#         active_set = _compute_active_set(self._Y_ineq[t], ineq, self._rho)
#         cost += jnp.dot(self._Y_ineq[t], ineq) + 0.5 * self._rho * jnp.dot(active_set * ineq, ineq) 

#         return cost
    
#     def gradient(self, x, u, t):
#         grad_x, grad_u = self._base_cost_model.gradient(x, u, t)
        
#         ineq = self._constraint_fn(x, u, t)
#         active_set = _compute_active_set(self._Y_ineq[t], ineq)

#         jac_ineq_x = jax.jacobian(self._constraint_fn, argnums=0)(x, u, t)  # (m, nx)
#         # jac_ineq_x = jac_ineq_x[:,:-1] # TODO: hardcoded

#         # jax.debug.print("t: {}, jac_ineq_x: {}\n", t, jac_ineq_x)

#         jac_ineq_u = jax.jacobian(self._constraint_fn, argnums=1)(x, u, t)  # (m, nu)
        
#         grad_x += jac_ineq_x.T @ self._Y_ineq[t] + self._rho * jac_ineq_x.T @ (active_set * ineq)
#         grad_u += jac_ineq_u.T @ self._Y_ineq[t] + self._rho * jac_ineq_u.T @ (active_set * ineq)
        
#         return grad_x, grad_u
    
#     def hessian(self, x, u, t):
#         # Y_eq = params["Y_eq"]
#         # Y_ineq = params["Y_ineq"]
#         # rho = params["rho"]
#         hess_xx, hess_uu, hess_xu = self._base_cost_model.hessian(x, u, t)

#         ineq = self._constraint_fn(x, u, t)
#         active_set = _compute_active_set(self._Y_ineq[t], ineq)

#         jac_ineq_x = jax.jacobian(self._constraint_fn, argnums=0)(x, u, t)  # (m, nx)
#         # jac_ineq_x = jac_ineq_x[:,:-1] # TODO: hardcoded

#         jac_ineq_u = jax.jacobian(self._constraint_fn, argnums=1)(x, u, t)  # (m, nu)

#         hess_xx += self._rho * jac_ineq_x.T @ (active_set[:, None] * jac_ineq_x)
#         hess_uu += self._rho * jac_ineq_u.T @ (active_set[:, None] * jac_ineq_u)

#         return hess_xx, hess_uu, hess_xu
    
#     def _tree_flatten(self):
#         """Flatten the object for JAX pytree registration."""
#         children = (self._Y_eq, self._Y_ineq, self._rho)
#         aux_data = (self._base_cost_model, self._constraint_fn)
#         return children, aux_data
    
#     @classmethod
#     def _tree_unflatten(cls, aux_data, children):
#         """Unflatten the object for JAX pytree registration."""
#         base_cost_model, constraint_fn = aux_data
#         Y_eq, Y_ineq, rho = children
#         return cls(base_cost_model, constraint_fn, Y_eq, Y_ineq, rho)


# # Register AugmentedLagrangian as a JAX pytree for JIT compatibility
# tree_util.register_pytree_node(
#     AugmentedLagrangian,
#     AugmentedLagrangian._tree_flatten,
#     AugmentedLagrangian._tree_unflatten
# )


# def _compute_active_set(y: jnp.ndarray, g: jnp.ndarray, rho: float) -> jnp.ndarray:
#     return (g + y / rho) > 0.0

# def _inequality_projection(y: jnp.ndarray, g: jnp.ndarray, rho: float) -> jnp.ndarray:
#     return jnp.maximum(g + y / rho, 0.0)

def _compute_active_set(dual_var: jnp.ndarray, constraint_val: jnp.ndarray) -> jnp.ndarray:
    return jnp.invert(jnp.isclose(dual_var, 0.0) & (constraint_val < 0.0))

class AugmentedLagrangian(AbstractCost):
    # TODO: support equality constraints
    def __init__(self, 
                 base_cost_model,
                 equality_constraint_model, 
                 inequality_constraint_model,
                 Y_eq, 
                 Y_ineq, 
                 rho):
        self._base_cost_model = base_cost_model
        self._equality_constraint_model = equality_constraint_model
        self._inequality_constraint_model = inequality_constraint_model
        self._Y_eq = Y_eq 
        self._Y_ineq = Y_ineq
        self._rho = rho

    def cost_fn(self, x, u, t):
        cost = self._base_cost_model.cost_fn(x, u, t)
        ineq = self._inequality_constraint_model.constraint_fn(x, u, t)

        active_set = _compute_active_set(self._Y_ineq[t], ineq)
        cost += jnp.dot(self._Y_ineq[t], ineq) + 0.5 * self._rho * jnp.dot(active_set * ineq, ineq) 

        return cost
    
    def gradient(self, x, u, t):
        grad_x, grad_u = self._base_cost_model.gradient(x, u, t)
        jac_ineq_x, jac_ineq_u = self._inequality_constraint_model.jacobian(x, u, t)
        
        ineq = self._inequality_constraint_model.constraint_fn(x, u, t)        
        active_set = _compute_active_set(self._Y_ineq[t], ineq)

        # grad_x += self._rho * jac_ineq_x.T @ ineq
        # grad_u += self._rho * jac_ineq_u.T @ ineq
        grad_x += jac_ineq_x.T @ self._Y_ineq[t] + self._rho * jac_ineq_x.T @ (active_set * ineq)
        grad_u += jac_ineq_u.T @ self._Y_ineq[t] + self._rho * jac_ineq_u.T @ (active_set * ineq)

        return grad_x, grad_u
    
    def hessian(self, x, u, t):
        hess_xx, hess_uu, hess_xu = self._base_cost_model.hessian(x, u, t)
        jac_ineq_x, jac_ineq_u = self._inequality_constraint_model.jacobian(x, u, t)

        ineq = self._inequality_constraint_model.constraint_fn(x, u, t)
        active_set = _compute_active_set(self._Y_ineq[t], ineq)

        hess_xx += self._rho * jac_ineq_x.T @ (active_set[:, None] * jac_ineq_x)
        hess_uu += self._rho * jac_ineq_u.T @ (active_set[:, None] * jac_ineq_u)

        return hess_xx, hess_uu, hess_xu
    
    def _tree_flatten(self):
        """Flatten the object for JAX pytree registration."""
        children = (self._Y_eq, self._Y_ineq, self._rho)
        aux_data = (self._base_cost_model, self._equality_constraint_model, self._inequality_constraint_model)
        return children, aux_data
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """Unflatten the object for JAX pytree registration."""
        base_cost_model, equality_constraint_model, inequality_constraint_model = aux_data
        Y_eq, Y_ineq, rho = children
        return cls(base_cost_model, equality_constraint_model, inequality_constraint_model, Y_eq, Y_ineq, rho)


# Register AugmentedLagrangian as a JAX pytree for JIT compatibility
tree_util.register_pytree_node(
    AugmentedLagrangian,
    AugmentedLagrangian._tree_flatten,
    AugmentedLagrangian._tree_unflatten
)
