import unittest
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.scipy as jsp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ilqrx.lqr import lqr_backward, lqr_forward, lqr_dual_variables, lqr_regularized_backward, lqr_regularized_forward

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

def form_KKT_system(Q, QT, R, q, qT, r, A, B, c, T, x0, regu):
    # Stack variables as follows:
    # [x_0, x_1, ..., x_{T-1}, x_T, u_0, u_1, ..., u_{T-1}]
    n = Q.shape[0]
    m = R.shape[0]
    P = jsp.linalg.block_diag(
        jnp.kron(jnp.eye(T), Q), QT, jnp.kron(jnp.eye(T), R))
    h = jnp.hstack(
        [jnp.kron(jnp.ones(T), q), qT, jnp.kron(jnp.ones(T), r)]
    )
    Ax = jnp.kron(jnp.eye(T + 1), -jnp.eye(n)) + jnp.kron(jnp.eye(T + 1, k=-1), A)
    Bu = jnp.vstack([jnp.zeros((n, T * m)), jnp.kron(jnp.eye(T), B)])
    Aeq = jnp.hstack([Ax, Bu])
    beq = jnp.hstack([-x0, jnp.kron(jnp.ones(T), -c)])

    regu_block = -regu * jnp.eye((T + 1) * n)
    regu_block = regu_block.at[:n, :n].set(0.0) # Do not regularize x0

    KKT_matrix = jnp.block([
        [P, Aeq.T],
        [Aeq, regu_block] 
    ])
    KKT_rhs = jnp.hstack([-h, beq])

    return KKT_matrix, KKT_rhs


class TestLQR(unittest.TestCase):
    def setUp(self):
        Ad = jnp.array([
        [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
        [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
        [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
        [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
        [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
        [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
        [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
        [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
        [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
        [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
        [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
        [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
        ])

        Bd = jnp.array([
        [0.,      -0.0726,  0.,     0.0726],
        [-0.0726,  0.,      0.0726, 0.    ],
        [-0.0152,  0.0152, -0.0152, 0.0152],
        [-0.,     -0.0006, -0.,     0.0006],
        [0.0006,   0.,     -0.0006, 0.0000],
        [0.0106,   0.0106,  0.0106, 0.0106],
        [0,       -1.4512,  0.,     1.4512],
        [-1.4512,  0.,      1.4512, 0.    ],
        [-0.3049,  0.3049, -0.3049, 0.3049],
        [-0.,     -0.0236,  0.,     0.0236],
        [0.0236,   0.,     -0.0236, 0.    ],
        [0.2107,   0.2107,  0.2107, 0.2107]])

        nx = Ad.shape[0]
        nu = Bd.shape[1]
        
        Q = jnp.diag(jnp.array([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.]))
        QT = 5 * Q
        R = 0.1 * jnp.eye(4)
        T = 20

        xr = jnp.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        q  = -(xr.T @ Q)
        qT = -(xr.T @ QT)
        r  = jnp.zeros(nu)
        M  = jnp.zeros((nu, nx))
        c  = jnp.ones(nx) * 0.01

        self.Qs = jnp.tile(Q, (T + 1, 1, 1))
        self.Qs = self.Qs.at[-1].set(QT)
        self.qs = jnp.tile(q, (T + 1, 1))    
        self.qs = self.qs.at[-1].set(qT)
        self.Rs = jnp.tile(R, (T, 1, 1))
        self.rs = jnp.tile(r, (T, 1))    
        self.Ms = jnp.tile(M, (T, 1, 1))
        self.As = jnp.tile(Ad, (T, 1, 1))
        self.Bs = jnp.tile(Bd, (T, 1, 1))
        self.cs = jnp.tile(c, (T, 1))
        
        self.x0 = jnp.zeros(12)
        self.nx = nx
        self.nu = nu
        self.T = T

        self.KKT_matrix1, self.KKT_rhs1 = form_KKT_system(
            Q, QT, R, q, qT, r, Ad, Bd, c, T, self.x0, regu=0.0)
        self.KKT_matrix2, self.KKT_rhs2 = form_KKT_system(
            Q, QT, R, q, qT, r, Ad, Bd, c, T, self.x0, regu=0.1)
        
    def test_lqr(self):
        nx = self.nx
        nu = self.nu
        T = self.T

        sol1 = jnp.linalg.solve(
            self.KKT_matrix1, self.KKT_rhs1)

        xs1 = sol1[: (T + 1) * nx].reshape((T + 1, nx))
        us1 = sol1[(T + 1) * nx : (T + 1) * nx + T * nu].reshape((T, nu))
        lams1 = sol1[(T + 1) * nx + T * nu :].reshape((T + 1, nx))

        Ks2, ks2, Ps2, ps2 = lqr_backward(self.Qs, self.qs, self.Rs, self.rs, self.Ms, self.As, self.Bs, self.cs)
        xs2, us2 = lqr_forward(self.x0, Ks2, ks2, self.As, self.Bs, self.cs)
        lams2 = lqr_dual_variables(Ps2, ps2, xs2)
        
        self.assertEqual(jnp.allclose(xs1, xs2), True)
        self.assertEqual(jnp.allclose(us1, us2), True)
        self.assertEqual(jnp.allclose(lams1, lams2), True)

    def test_regularized_lqr(self):
        nx = self.nx
        nu = self.nu
        T = self.T

        sol1 = jnp.linalg.solve(
            self.KKT_matrix2, self.KKT_rhs2)

        xs1 = sol1[: (T + 1) * nx].reshape((T + 1, nx))
        us1 = sol1[(T + 1) * nx : (T + 1) * nx + T * nu].reshape((T, nu))
        lams1 = sol1[(T + 1) * nx + T * nu :].reshape((T + 1, nx))

        Ks2, ks2, Ps2, ps2, Us2 = lqr_regularized_backward(
            self.Qs, self.qs, self.Rs, self.rs, self.Ms, self.As, self.Bs, self.cs, 0.1)
        xs2, us2 = lqr_regularized_forward(self.x0, Ks2, ks2, self.As, self.Bs, self.cs, Us2, ps2, 0.1)
        lams2 = lqr_dual_variables(Ps2, ps2, xs2)

        self.assertEqual(jnp.allclose(xs1, xs2), True)
        self.assertEqual(jnp.allclose(us1, us2), True)
        self.assertEqual(jnp.allclose(lams1, lams2), True)


if __name__ == '__main__':
    unittest.main()