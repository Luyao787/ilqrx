import jax
from jax import lax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Optional, Tuple
from ilqrx.linesearch import serial_linesearch_al
from ilqrx.ocp import OptimalControlProblem
from ilqrx.utils import pad, compute_expected_cost_reduction, regularize, vectorize, linearize, quadratize, hamilton, traj_cost, compute_dynamics_jacobians, compute_defects, compute_norm, vectorize
from ilqrx.lqr import lqr_backward, lqr_forward, lqr_dual_variables
from ilqrx.type import ILQRSolution, ILQRStatus, CILQRSolution, CILQRStatus, CILQRSolverOptions
from modeling.costs.augmented_lagrangian import AugmentedLagrangian

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
    if dynamic_model.use_quaternion == False:
        # Rely on Automatic Differentiation
        cost_gradients = linearize(cost_model.cost_fn, argnums=3)        
        # cost_gradients = vectorize(cost_model.gradient, argnums=3)
        cost_hessians = quadratize(hamilton(cost_model.cost_fn, dynamic_model.dynamics_dt), argnums=4)
    else:
        cost_gradients = vectorize(cost_model.gradient, argnums=3)
        cost_hessians = vectorize(cost_model.hessian, argnums=3)
        

    dynamics_jacobians = partial(compute_dynamics_jacobians, dynamic_model.jacobian_dt)
    state_update_mapped = jax.vmap(dynamic_model.state_update, in_axes=(0, 0))

    def get_lqr_params(X, U, Y_dyn):
        timesteps = jnp.arange(X.shape[0])              
        if dynamic_model.use_quaternion == False:
            Qs, Rs, Ms = cost_hessians(X, pad(U), timesteps, pad(Y_dyn[1:]))
        else:
            Qs, Rs, Ms = cost_hessians(X, pad(U), timesteps)
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


class CILQRSolver:
    """
    Constrained Iterative Linear Quadratic Regulator (CILQR) solver.
    """
    def __init__(self, 
                 problem: OptimalControlProblem,
                 options: Optional[CILQRSolverOptions] = None):
        
        self._problem = problem
        self._options = options if options is not None else CILQRSolverOptions()

    def solve(self, 
              X_init: jnp.ndarray, 
              U_init: jnp.ndarray,
              verbose: bool = False) -> CILQRSolution:
        
        self._validate_inputs(X_init, U_init)
        
        self._eval_equality_constraints = vectorize(self._problem._equality_constraint_model.constraint_fn, argnums=3)
        self._eval_inequality_constraints = vectorize(self._problem._inequality_constraint_model.constraint_fn, argnums=3)
        
        return self._solve_impl(X_init, U_init, verbose)

    def _solve_impl(self, 
                    X_init: jnp.ndarray, 
                    U_init: jnp.ndarray,
                    verbose: bool = False) -> CILQRSolution:
        """
        Core CILQR algorithm implementation.
        """   
        def cilqr_iteration(inputs: Tuple) -> Tuple:
            """Single CILQR iteration: solve iLQR subproblem and update dual variables."""
            (cilqr_solution, rho, opti_tol_temp, constr_vio_tol_temp) = inputs  
            if verbose:
                jax.debug.print("┌─ CILQR Iteration (iter={}, rho={:.2e})", cilqr_solution.num_iter, rho)
                            
            ilqr_solution = ilqr_solve(
                AugmentedLagrangian(
                    base_cost_model=self._problem.cost_model,
                    equality_constraint_model=self._problem._equality_constraint_model,
                    inequality_constraint_model=self._problem._inequality_constraint_model,
                    Y_eq=cilqr_solution.Y_eq,
                    Y_ineq=cilqr_solution.Y_ineq,
                    rho=rho
                ),
                self._problem.dynamic_model,
                cilqr_solution.X,
                cilqr_solution.U,
                cilqr_solution.Y_dyn,
                opti_tol=opti_tol_temp,
                max_iter=self._options.inner_ilqr_max_iter,
                verbose=verbose,
            )

            # Evaluate constraints at current solution
            U_pad = pad(ilqr_solution.U)
            timesteps = jnp.arange(ilqr_solution.X.shape[0])
            equality_constraints = self._eval_equality_constraints(ilqr_solution.X, U_pad, timesteps)
            inequality_constraints = self._eval_inequality_constraints(ilqr_solution.X, U_pad, timesteps)
            
            # Compute constraint violation
            ineq_constr_vio = jnp.maximum(inequality_constraints, -cilqr_solution.Y_ineq / rho)
            max_constr_vio = jnp.maximum(
                jnp.max(jnp.abs(equality_constraints)), jnp.max(jnp.abs(ineq_constr_vio))
            )

            (Y_eq, Y_ineq, rho, opti_tol_temp, constr_vio_tol_temp) = _adaptive_dual_update(
                equality_constraints, 
                inequality_constraints,
                cilqr_solution.Y_eq, 
                cilqr_solution.Y_ineq,
                rho,
                opti_tol_temp, 
                constr_vio_tol_temp,
                max_constr_vio,
                self._options,
                verbose
            )
                
            # max_constr_vio = jnp.maximum(max_constr_vio, ilqr_solution.max_dyn_vio)
            iter_lqr = cilqr_solution.num_iter + ilqr_solution.num_iter
            
            if verbose:
                jax.debug.print("└─ iLQR status: {}, constraint violation: {:.2e}, opti_tol: {:.2e}, constr_vio_tol: {:.2e}",
                        ilqr_solution.status, max_constr_vio, opti_tol_temp, constr_vio_tol_temp)

            return (
                CILQRSolution(
                    X=ilqr_solution.X,
                    U=ilqr_solution.U,
                    Y_dyn=ilqr_solution.Y_dyn,
                    Y_eq=Y_eq,
                    Y_ineq=Y_ineq,
                    cost=ilqr_solution.cost,
                    num_iter=iter_lqr,
                    max_constr_vio=max_constr_vio,
                    opti_res_infnorm=ilqr_solution.kkt_res_infnorm,
                    cilqr_status=CILQRStatus.MAX_ITER_REACHED,  # Placeholder, will be updated later
                    ilqr_status=ilqr_solution.status,
                ), 
                rho, opti_tol_temp, constr_vio_tol_temp)
        
        def continuation_criterion(inputs: Tuple) -> bool:
            """Check if CILQR should continue iterating."""
            (cilqr_solution, _, _, _) = inputs

            not_converged = jnp.logical_or(
                cilqr_solution.opti_res_infnorm > self._options.opti_tol,
                cilqr_solution.max_constr_vio > self._options.constr_vio_tol)
            ilqr_ok = cilqr_solution.ilqr_status == ILQRStatus.CONVERGED
            
            return jnp.logical_and(
                jnp.logical_and(not_converged, cilqr_solution.num_iter < self._options.max_iter), 
                ilqr_ok)
        

        T = U_init.shape[0]
        nx = self._problem.dynamic_model.state_dim
        ndx = self._problem.dynamic_model.tangent_space_dim
        
        U_pad = pad(U_init)
        timesteps = jnp.arange(T + 1)
        equality_constraints = self._eval_equality_constraints(X_init, U_pad, timesteps)
        inequality_constraints = self._eval_inequality_constraints(X_init, U_pad, timesteps)
        
        Y_dyn = jnp.zeros((T+1, ndx)) # Dual variables for dynamics
        Y_eq = jnp.zeros_like(equality_constraints) # Dual variables for equality constraints
        Y_ineq = jnp.zeros_like(inequality_constraints) # Dual variables for inequality constraints
        
        rho = self._options.rho_init
            
        opti_tol_temp = self._options.opti_tol_init
        constr_vio_tol_temp = self._options.constr_vio_tol_init
        
        # Main CILQR loop
        if verbose:
            jax.debug.print("\n╔═══ Starting CILQR Optimization ═══╗")
        
        inputs = (
            CILQRSolution(
                X=X_init,
                U=U_init,
                Y_dyn=Y_dyn,
                Y_eq=Y_eq,
                Y_ineq=Y_ineq,
                cost=jnp.inf,
                num_iter=0,
                max_constr_vio=jnp.inf,
                opti_res_infnorm=jnp.inf,
                cilqr_status=CILQRStatus.CONVERGED,
                ilqr_status=ILQRStatus.CONVERGED
            ),
            rho,
            opti_tol_temp,
            constr_vio_tol_temp,
        )
        outputs = jax.lax.while_loop(continuation_criterion, cilqr_iteration, inputs)

        (cilqr_solution,
         rho,
         opti_tol_temp,
         constr_vio_tol_temp) = outputs
        
        # Determine final status
        cilqr_status = lax.select(
            cilqr_solution.ilqr_status != ILQRStatus.CONVERGED,
            CILQRStatus.ILQR_FAILURE,
            lax.select(
                cilqr_solution.max_constr_vio <= self._options.constr_vio_tol,
                CILQRStatus.CONVERGED,
                CILQRStatus.MAX_ITER_REACHED
            )
        )

        if verbose:
            jax.debug.print("╚═══ CILQR Finished (status={}, iters={}, constraint violation={:.2e}) ═══╝\n",
                          cilqr_status, cilqr_solution.num_iter, cilqr_solution.max_constr_vio)
            
        cilqr_solution = cilqr_solution._replace(cilqr_status=cilqr_status)
        
        return cilqr_solution   
    
    def _validate_inputs(self, X: jnp.ndarray, U: jnp.ndarray) -> None:
        if X.ndim != 2 or U.ndim != 2:
            raise ValueError(f"X and U must be 2D arrays, got shapes {X.shape} and {U.shape}")
        if X.shape[0] != U.shape[0] + 1:
            raise ValueError(f"X must have one more stage than U, got {X.shape[0]} vs {U.shape[0]}")
        if X.shape[0] < 2:
            raise ValueError(f"Trajectory must have at least 2 stages, got {X.shape[0]}")
        


# --- Helper functions for dual updates and projections ---

def _dual_update(constraint: jnp.ndarray, dual: jnp.ndarray, rho: float) -> jnp.ndarray:
    """Standard augmented Lagrangian dual update."""
    return dual + rho * constraint


def _inequality_projection(x: jnp.ndarray) -> jnp.ndarray:
    """Project dual variables for inequality constraints (non-negativity)."""
    return jnp.maximum(x, 0.0)


def _adaptive_dual_update(
    equality_constraint_vals: jnp.ndarray,
    inequality_constraint_vals: jnp.ndarray,
    Y_eq: jnp.ndarray,
    Y_ineq: jnp.ndarray,
    rho: float,
    opti_tol_temp: float,
    constr_vio_tol_temp: float,
    max_constr_vio: float,
    solver_options: CILQRSolverOptions,
    verbose: bool
) -> Tuple:
    """
    Adaptive dual variable and penalty parameter update (BCL method).
    
    If constraint violation is below tolerance, update duals and tighten tolerances (success).
    Otherwise, increase penalty and relax tolerances (failure).
    """

    def success_update(inputs):
        """Update when constraint violation tolerance is satisfied."""
        if verbose:
            jax.debug.print("  Constraint violation within tolerance. Tightening tolerances.")
        
        rho_, opti_tol_, constr_vio_tol_, Y_eq, Y_ineq = inputs

        Y_eq = _dual_update(equality_constraint_vals, Y_eq, rho_)
        Y_ineq = _dual_update(inequality_constraint_vals, Y_ineq, rho_)
        Y_ineq = _inequality_projection(Y_ineq)

        # Tighten tolerances (success)
        opti_tol_ *= (1 / rho_) ** solver_options.bcl_beta_opti
        constr_vio_tol_ *= (1 / rho_) ** solver_options.bcl_beta_constr_vio
        
        return rho_, opti_tol_, constr_vio_tol_, Y_eq, Y_ineq
    
    def failure_update(inputs):
        """Update when constraint violation tolerance is not satisfied."""
        if verbose:
            jax.debug.print("  Constraint violation not within tolerance. Increasing penalty weight")
        
        rho_, opti_tol_, constr_vio_tol_, Y_eq, Y_ineq = inputs

        # Increase penalty parameter
        rho_ = jnp.minimum(solver_options.rho_max, rho_ * solver_options.rho_scaling)
        # Relax tolerances (failure)
        opti_tol_ = solver_options.opti_tol_init * (1 / rho_) ** solver_options.bcl_alpha_opti
        constr_vio_tol_ = solver_options.constr_vio_tol_init * (1 / rho_) ** solver_options.bcl_alpha_constr_vio
        
        return rho_, opti_tol_, constr_vio_tol_, Y_eq, Y_ineq

    # Choose update strategy based on constraint violation
    rho, opti_tol_temp, constr_vio_tol_temp, Y_eq, Y_ineq = lax.cond(
        max_constr_vio <= constr_vio_tol_temp,
        success_update,
        failure_update,
        (rho, opti_tol_temp, constr_vio_tol_temp, Y_eq, Y_ineq)
    )

    # Enforce minimum tolerances
    opti_tol_temp = jnp.maximum(opti_tol_temp, solver_options.opti_tol)
    constr_vio_tol_temp = jnp.maximum(constr_vio_tol_temp, solver_options.constr_vio_tol)
    
    return (Y_eq, Y_ineq,
            rho, 
            opti_tol_temp, constr_vio_tol_temp)