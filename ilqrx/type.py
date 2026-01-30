import jax.numpy as jnp
import enum
from typing import NamedTuple

# --- ILQR related types ---

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

# --- CILQR related types ---

class CILQRStatus(enum.IntEnum):
    CONVERGED = 0
    ILQR_FAILURE = -1
    MAX_ITER_REACHED = 1


class CILQRSolution(NamedTuple):
    X: jnp.ndarray  # Optimized state trajectory (T+1, nx)
    U: jnp.ndarray  # Optimized control trajectory (T, nu)
    Y_dyn: jnp.ndarray  # Dual variables for dynamics (T, nx)
    Y_eq: jnp.ndarray  # Dual variables for equality constraints
    Y_ineq: jnp.ndarray  # Dual variables for inequality constraints
    cost: float  # Final cost value
    num_iter: int  # Total number of iLQR iterations
    max_constr_vio: float  # Maximum constraint violation
    opti_res_infnorm: float  # Infinity norm of the optimality residual
    cilqr_status: CILQRStatus
    ilqr_status: ILQRStatus


class CILQRSolverOptions(NamedTuple):
    # Convergence criteria
    max_iter: int = 500  # Maximum total iLQR iterations
    inner_ilqr_max_iter: int = 500  # Maximum iLQR iterations per CILQR iteration
    opti_tol: float = 1e-5  # Optimality tolerance (gradient norm)
    constr_vio_tol: float = 1e-5  # Constraint violation tolerance
    
    # Adaptive tolerance scheduling (BCL method)
    opti_tol_init: float = 1.0  # Initial optimality tolerance
    constr_vio_tol_init: float = 1.0  # Initial constraint violation tolerance
    bcl_alpha_opti: float = 1.0  # Optimality tolerance scaling (failure)
    bcl_alpha_constr_vio: float = 0.1  # Constraint tolerance scaling (failure)
    bcl_beta_opti: float = 1.0  # Optimality tolerance scaling (success)
    bcl_beta_constr_vio: float = 0.9  # Constraint tolerance scaling (success)
    
    # Penalty parameters
    rho_init: float = 10.0  # Initial penalty parameter for constraints
    # rho_dyn_init: float = 1e5  # Initial penalty parameter for dynamics
    rho_scaling: float = 100.0  # Penalty parameter scaling factor
    rho_max: float = 1e10  # Maximum penalty parameter
    # rho_dyn_max: float = 1e8 # Maximum penalty parameter for dynamics
    # rho_dyn_max: float = 1e10 # Maximum penalty parameter for dynamics

    # use_dual_regularization_for_dynamics: bool = False