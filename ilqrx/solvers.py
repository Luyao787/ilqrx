import jax
from jax import lax
import jax.numpy as jnp
from functools import partial
import enum
from typing import NamedTuple
from ilqrx.linesearch import serial_linesearch_al
from ilqrx.utils import pad, compute_expected_cost_reduction, regularize, linearize, quadratize, lagrangian, traj_cost, compute_dynamics_jacobians, compute_defects, compute_norm
from ilqrx.lqr import lqr_backward, lqr_forward, lqr_dual_variables

class ILQRStatus(enum.IntEnum):
    CONVERGED = 0
    LINE_SEARCH_FAILURE = -1
    MAX_ITER_REACHED = 1

class ILQRSolution(NamedTuple):
    X: jnp.ndarray
    U: jnp.ndarray
    Y_dyn: jnp.ndarray
    cost: float
    num_iter: int
    status: ILQRStatus
    kkt_res_infnorm: float

def ilqr_solve(
    cost_model,
    dynamic_model,
    X,
    U,
    Y_dyn,
    opti_tol=1e-5,
    max_iter=500,
    verbose=False,
):
    cost_gradients = linearize(cost_model.cost_fn, argnums=3)
    # cost_hessians = quadratize(cost_model.cost_fn, argnums=3)
    cost_hessians = quadratize(lagrangian(cost_model.cost_fn, dynamic_model.dynamics_dt, X[0]), argnums=5)

    dynamics_jacobians = partial(compute_dynamics_jacobians, dynamic_model.jacobian_dt)
    state_update_mapped = jax.vmap(dynamic_model.state_update, in_axes=(0, 0))

    def get_lqr_params(X, U, Y_dyn):
        timesteps = jnp.arange(X.shape[0])              
        Qs, Rs, Ms = cost_hessians(X, pad(U), timesteps, pad(Y_dyn[1:]), Y_dyn)
        Qs, Rs = regularize(Qs, Rs, Ms, psd_delta=1e-6)
        qs, rs = cost_gradients(X, pad(U), timesteps)
        As, Bs = dynamics_jacobians(X, U)
        cs = compute_defects(dynamic_model.dynamics_defect, X, U)
        Rs = Rs[:-1, ...]
        rs = rs[:-1, ...]
        Ms = Ms[:-1, ...]
        Ms = Ms.swapaxes(-2, -1)  # TODO:
        return Qs, Rs, Ms, qs, rs, As, Bs, cs
    
    def compute_search_direction(X, U, Y_dyn):
        lqr_params = get_lqr_params(X, U, Y_dyn)
        Qs, Rs, Ms, qs, rs, As, Bs, cs = lqr_params
        Ks, ks, Ps, ps, = lqr_backward(Qs, qs, Rs, rs, Ms, As, Bs, cs)
        delta_X, delta_U = lqr_forward(jnp.zeros(dynamic_model.tangent_space_dim), Ks, ks, As, Bs, cs) 
        Y_dyn_trial = lqr_dual_variables(Ps, ps, delta_X)

        delta_Y_dyn = Y_dyn_trial - Y_dyn
        # X_trial = X + delta_X
        X_trial = state_update_mapped(X, delta_X)
        U_trial = U + delta_U

        res_x, res_u, defects = compute_kkt_residuals(X_trial, U_trial, Y_dyn_trial)
        res_x_infnorm = compute_norm(res_x, order=jnp.inf)
        res_u_infnorm = compute_norm(res_u, order=jnp.inf)
        defect_infnorm = compute_norm(defects, order=jnp.inf)
        kkt_res_infnorm = jnp.maximum(jnp.maximum(res_x_infnorm, res_u_infnorm), defect_infnorm)

        return delta_X, delta_U, delta_Y_dyn, res_x_infnorm, res_u_infnorm, defect_infnorm, kkt_res_infnorm, lqr_params
    
    def compute_kkt_residuals(X, U, Y_dyn):
        timesteps = jnp.arange(X.shape[0])
        qs, rs = cost_gradients(X, pad(U), timesteps)
        As, Bs = dynamics_jacobians(X, U)

        # KKT condition
        res_u_stage = lambda r, lam_next, B: r + B.T @ lam_next
        res_x_stage = lambda q, lam, lam_next, A: q - lam + A.T @ lam_next
        res_x_terminal = lambda q, lam: q - lam
        res_u_mapped = jax.vmap(res_u_stage)
        res_x_mapped = jax.vmap(res_x_stage)

        res_u = res_u_mapped(
            rs[:-1],
            Y_dyn[1:],
            Bs
        )
        res_x = jnp.concatenate([
            res_x_mapped(
                qs[1:-1], 
                Y_dyn[1:-1],
                Y_dyn[2:],
                As[1:]
            ),
            res_x_terminal(
                qs[-1],
                Y_dyn[-1]
            )[None, :]   
        ], axis=0)

        defects = compute_defects(dynamic_model.dynamics_defect, X, U)

        return res_x, res_u, defects

    

    def body(inputs):
        (X, U, Y_dyn,
         _, _, iter_id, _, rho_dyn) = inputs

        # compute the new search direction
        (delta_X, delta_U, delta_Y_dyn, 
         res_x_infnorm, res_u_infnorm, defect_infnorm, kkt_res_infnorm, 
         lqr_params) = compute_search_direction(X, U, Y_dyn)
        Qs, Rs, Ms, _, _, _, _, _ = lqr_params

        skip_line_search_flag = kkt_res_infnorm <= jnp.minimum(opti_tol, 1e-5) # TODO: Remove magic number

        def run_line_search():
            quad_redu = compute_expected_cost_reduction(delta_X, delta_U, Qs, Rs, Ms)
            return serial_linesearch_al(
                cost_model.cost_fn,
                cost_gradients,
                dynamic_model.dynamics_defect,
                dynamics_jacobians,
                state_update_mapped,
                X, U, Y_dyn,
                delta_X, delta_U, delta_Y_dyn,
                rho_dyn,
                quad_redu,
            )

        def skip_line_search():
            X_new = state_update_mapped(X, delta_X)
            U_new = U + delta_U
            Y_dyn_new = Y_dyn + delta_Y_dyn
            cost_new = traj_cost(cost_model.cost_fn, X_new, U_new)
            return X_new, U_new, Y_dyn_new, cost_new, 1.0, rho_dyn


        X_new, U_new, Y_dyn_new, cost_new, alpha, rho_dyn = lax.cond(
            skip_line_search_flag,
            skip_line_search,
            run_line_search
        )

        iter_id += 1

        if verbose:
            jax.debug.print(
                "{i:^4d} | {c:^16.6e} | {lag_x:^16.6e} | {lag_u:^16.6e} | {defect:^17.6e} | {a:^16.6e}",
                i=iter_id, c=cost_new, lag_x=res_x_infnorm, lag_u=res_u_infnorm, defect=defect_infnorm, a=alpha
            )

        return (X_new, U_new, Y_dyn_new, 
                cost_new, alpha, iter_id, kkt_res_infnorm, rho_dyn)
    
    def continuation_criterion(inputs):
        (_, _, _, 
         _, alpha, iter_id, kkt_res_infnorm, _) = inputs
        still_continue = jnp.logical_and(
            iter_id < max_iter,
            kkt_res_infnorm > opti_tol)
        return jnp.logical_and(still_continue, alpha > 0.)


    cost = jnp.inf
    alpha = 1.0
    iter_id = 0
    kkt_res_infnorm = jnp.inf
    rho_dyn = 0.

    if verbose:
        jax.debug.print("---\niter |      cost        |  lag_x_inf_norm  |  lag_u_inf_norm  |  defect_inf_norm  |      alpha")
    
    inputs = (X, U, Y_dyn,
              cost, alpha, iter_id, kkt_res_infnorm, rho_dyn)
    (X, U, Y_dyn,
     cost, alpha, iter_id, kkt_res_infnorm, _) = lax.while_loop(
        continuation_criterion,
        body,
        inputs
    )

    sol_status = lax.cond(
        kkt_res_infnorm <= opti_tol,
        lambda: ILQRStatus.CONVERGED,
        lambda: lax.cond(
            alpha <= 0.,
            lambda: ILQRStatus.LINE_SEARCH_FAILURE,
            lambda: lax.cond(
                iter_id >= max_iter,
                lambda: ILQRStatus.MAX_ITER_REACHED,
                lambda: ILQRStatus.CONVERGED  # Should not reach here given loop conditions
            )
        )
    )
        
    return ILQRSolution(
        X=X,
        U=U,
        Y_dyn=Y_dyn,
        cost=cost,
        num_iter=iter_id,
        status=sol_status,
        kkt_res_infnorm=kkt_res_infnorm,
    )