import jax
import jax.lax as lax
import jax.numpy as jnp
from ilqrx.utils import pad, traj_cost

def serial_linesearch_al(
    cost_fn,
    cost_gradients,
    compute_defect,
    dynamics_jacobians,
    state_update_mapped,
    X, U, Y_dyn, 
    delta_X, delta_U, delta_Y_dyn,
    mu_dyn_prev,
    quad_redu,
    max_ls_iters=20,
    ls_decay_rate=0.5, 
    beta=0.1
):
    def merit_fn(alpha, mu_dyn):
        def stage_merit(x, u, t, x_next, y_dyn, mu_dyn):
            defect = compute_defect(x, u, t, x_next)
            stage_merit = cost_fn(x, u, t)
            stage_merit += jnp.dot(y_dyn, defect) + 0.5 * jnp.dot(defect, mu_dyn * defect)
            return stage_merit
        
        X_new = state_update_mapped(X, alpha * delta_X)
        U_new = U + alpha * delta_U
        Y_dyn_new = Y_dyn + alpha * delta_Y_dyn
        horizon = X.shape[0] - 1       
        timesteps = jnp.arange(horizon)
        stage_merits = jax.vmap(stage_merit, in_axes=(0, 0, 0, 0, 0, None))(
        # stage_merits = jax.vmap(stage_merit, in_axes=(0, 0, 0, 0, 0, 0))(
            X_new[:-1], 
            U_new, 
            timesteps, 
            X_new[1:], 
            Y_dyn_new[1:], 
            mu_dyn
        )
        terminal_merit = cost_fn(X_new[-1], jnp.zeros_like(U_new[0]), horizon)
        total_merit = jnp.sum(stage_merits) + terminal_merit
        return total_merit
    
    def merit_derivative_fn(alpha, mu_dyn):
        def stage_merit_derivative(x, u, t, x_next, y_dyn, mu_dyn, dx, du, dx_next, dy_dyn, q, r, A, B):
            lin_dyn_res = jnp.where(t == 0, B @ du - dx_next, A @ dx + B @ du - dx_next)
            defect = compute_defect(x, u, t, x_next)
            stage_merit_derivative  = jnp.dot(q, dx) + jnp.dot(r, du)
            stage_merit_derivative += jnp.dot(dy_dyn, defect)           
            stage_merit_derivative += jnp.dot(y_dyn, lin_dyn_res)
            stage_merit_derivative += jnp.dot(lin_dyn_res, mu_dyn * defect)
            return stage_merit_derivative
        
        X_new = state_update_mapped(X, alpha * delta_X)
        U_new = U + alpha * delta_U
        Y_dyn_new = Y_dyn + alpha * delta_Y_dyn
        horizon = X.shape[0] - 1       
        timesteps = jnp.arange(horizon + 1)
        qs, rs = cost_gradients(X_new, pad(U_new), timesteps)
        As, Bs = dynamics_jacobians(X_new, U_new)
        stage_merit_derivatives = jax.vmap(stage_merit_derivative, in_axes=(0, 0, 0, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0))(
        # stage_merit_derivatives = jax.vmap(stage_merit_derivative, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
            X_new[:-1], 
            U_new, 
            timesteps[:-1], 
            X_new[1:], 
            Y_dyn_new[1:], 
            mu_dyn,
            delta_X[:-1],
            delta_U,
            delta_X[1:],
            delta_Y_dyn[1:],
            qs[:-1],
            rs[:-1],
            As,
            Bs,
        )
        terminal_merit_derivative = jnp.dot(qs[-1], delta_X[-1])
        total_merit_derivative = jnp.sum(stage_merit_derivatives) + terminal_merit_derivative
        return total_merit_derivative
    
    def body(inputs):
        _, _, _, _, alpha, iter_id, _, phi0, dphi0, mu_dyn = inputs
        
        merit_new = merit_fn(alpha, mu_dyn)
        sufficent_decrease_flag = merit_new <= phi0 + beta * alpha * dphi0

        # jax.debug.print("Line search iter: {iter}, alpha: {alpha}, merit_new: {merit_new}, phi0: {phi0}, dphi0: {dphi0}, sufficient decrease: {sufficent_decrease_flag}",
        #                 iter=iter_id, alpha=alpha, merit_new=merit_new, phi0=phi0, dphi0=dphi0, sufficent_decrease_flag=sufficent_decrease_flag)
        
        # Only update if the merit is reduced sufficiently
        X_return = jnp.where(sufficent_decrease_flag, state_update_mapped(X, alpha * delta_X), X)
        U_return = jnp.where(sufficent_decrease_flag, U + alpha * delta_U, U)
        Y_dyn_return = jnp.where(sufficent_decrease_flag, Y_dyn + alpha * delta_Y_dyn, Y_dyn)
        cost_return = jnp.where(sufficent_decrease_flag, traj_cost(cost_fn, X_return, U_return), traj_cost(cost_fn, X, U))
        
        alpha *= ls_decay_rate
        iter_id += 1
        
        return X_return, U_return, Y_dyn_return, cost_return, alpha, iter_id, sufficent_decrease_flag, phi0, dphi0, mu_dyn
    
    def condition(inputs):
        _, _, _, _, _, iter_id, sufficent_decrease_flag, _, _, _= inputs
        return jnp.logical_and(iter_id < max_ls_iters, sufficent_decrease_flag == False)
    
    def update_penalty_weight(mu_dyn_prev, dphi_0):
        
        # jax.debug.print("Updating penalty weight, dphi_0: {dphi_0}, quad_redu: {quad_redu}, mu_dyn_prev: {mu_dyn_prev}", 
                        # dphi_0=dphi_0, quad_redu=quad_redu, mu_dyn_prev=mu_dyn_prev)
        
        timesteps = jnp.arange(X.shape[0] - 1)
        defects = jax.vmap(compute_defect)(X[:-1], U, timesteps, X[1:])
        defects_flat = defects.flatten()
        delta_Y_dyn_flat = delta_Y_dyn[1:].flatten()
        
        mu_dyn_hat =  2 * jnp.linalg.norm(delta_Y_dyn_flat, ord=2) / jnp.linalg.norm(defects_flat, ord=2)
        mu_dyn = jnp.where(dphi_0 < -quad_redu, # do not forget the negative sign here
                           mu_dyn_prev,
                           jnp.maximum(mu_dyn_hat, 2 * mu_dyn_prev))
    
        return mu_dyn

    dphi0 = merit_derivative_fn(0.0, mu_dyn_prev)
    mu_dyn = update_penalty_weight(mu_dyn_prev, dphi0)
    
    # jax.debug.print("mu: {}, mu_prev:{}", mu_dyn, mu_dyn_prev)

    phi0 = merit_fn(0.0, mu_dyn)
    dphi0 = merit_derivative_fn(0.0, mu_dyn)
    
    alpha = 1.0
    iter_id = 0
    cost = jnp.inf
    sufficent_decrease_flag = False
    
    X, U, Y_dyn, cost, alpha, iter_id, _, _, _, mu_dyn = lax.while_loop(
        condition,
        body,
        (X, U, Y_dyn, cost, alpha, iter_id, sufficent_decrease_flag, phi0, dphi0, mu_dyn)
    )

    # Set alpha to 0.0 if line search failed
    alpha = jnp.where(iter_id >= max_ls_iters, 0.0, alpha / ls_decay_rate)

    return X, U, Y_dyn, cost, alpha, mu_dyn
