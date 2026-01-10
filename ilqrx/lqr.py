import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp

@jax.jit
def lqr_step(Q, R, S, q, r, P, p, A, B, c):
    
    symmetrize = lambda x: (x + x.T) / 2
 
    AtP = jnp.matmul(A.T, P)
    AtPA = symmetrize(jnp.matmul(AtP, A))
    BtP = jnp.matmul(B.T, P)
    BtPA = jnp.matmul(BtP, A)
    
    Qxx = Q + AtPA
    Quu = symmetrize(R + jnp.matmul(BtP, B))
    Qux = S + BtPA  # u^T @ S @ x
    
    Qx  = q + jnp.matmul(A.T, p) + jnp.matmul(AtP, c)
    Qu  = r + jnp.matmul(B.T, p) + jnp.matmul(BtP, c)
        
    chofac = jsp.linalg.cho_factor(Quu)
    K = -jsp.linalg.cho_solve(chofac, Qux)
    k = -jsp.linalg.cho_solve(chofac, Qu).squeeze()
    
    P = symmetrize(Qxx + Qux.T @ K)
    p = Qx + jnp.matmul(Qux.T, k)

    return K, k, P, p

@jax.jit
def lqr_backward(Qs, qs, Rs, rs, Ss, As, Bs, cs):
    
    T = Qs.shape[0] - 1
    n = Qs.shape[1]
    m = Rs.shape[1]

    Ps = jnp.zeros((T + 1, n, n))
    ps = jnp.zeros((T + 1, n))
    Ks = jnp.zeros((T, m, n))
    ks = jnp.zeros((T, m))

    Ps = Ps.at[-1].set(Qs[T])
    ps = ps.at[-1].set(qs[T])
    
    def body(tt, inputs):
        Ks, ks, Ps, ps = inputs         
        t = T - tt - 1
        K_t, k_t, P_t, p_t = lqr_step(Qs[t], Rs[t], Ss[t], qs[t], rs[t], Ps[t+1], ps[t+1], As[t], Bs[t], cs[t])
        
        Ks = Ks.at[t].set(K_t)
        ks = ks.at[t].set(k_t)
        Ps = Ps.at[t].set(P_t)
        ps = ps.at[t].set(p_t)
        
        return Ks, ks, Ps, ps
        
    return lax.fori_loop(0, T, body, (Ks, ks, Ps, ps))  

@jax.jit
def lqr_forward(x0, Ks, ks, As, Bs, cs):
    T, m, n = Ks.shape
    xs = jnp.zeros((T + 1, n))
    us = jnp.zeros((T, m))
    xs = xs.at[0].set(x0)
    
    def body(t, inputs):
        xs, us = inputs
        u_t = Ks[t] @ xs[t] + ks[t]
        x_t1 = As[t] @ xs[t] + Bs[t] @ u_t + cs[t]        
        us = us.at[t].set(u_t)
        xs = xs.at[t+1].set(x_t1)

        return xs, us
    
    return lax.fori_loop(0, T, body, (xs, us))

@jax.jit
def lqr_dual_variables(Ps, ps, xs):
    lams = jax.vmap(lambda P, p, x: P @ x + p)(Ps, ps, xs)
    return lams




@jax.jit
def lqr_regularized_step(Q, R, S, q, r, P, p, A, B, c, U, mu):
    
    symmetrize = lambda x: (x + x.T) / 2

    nx = Q.shape[0]
    I_nx = jnp.eye(nx)
    chofac_G = (U, False) 
    P_hat = jsp.linalg.cho_solve(chofac_G, P)
    p_hat = jsp.linalg.cho_solve(chofac_G, P @ c + p)

    AtP = jnp.matmul(A.T, P_hat)
    AtPA = jnp.matmul(AtP, A)
    BtP = jnp.matmul(B.T, P_hat)
    BtPA = jnp.matmul(BtP, A)
    
    Qxx = Q + AtPA
    Quu = R + jnp.matmul(BtP, B)
    Qux = S + BtPA  # u^T @ S @ x
    
    Qx  = q + jnp.matmul(A.T, p_hat)
    Qu  = r + jnp.matmul(B.T, p_hat)
        
    chofac = jsp.linalg.cho_factor(Quu)
    K = -jsp.linalg.cho_solve(chofac, Qux)
    k = -jsp.linalg.cho_solve(chofac, Qu).squeeze()
    
    P = symmetrize(Qxx + Qux.T @ K)
    p = Qx + jnp.matmul(Qux.T, k)

    G = I_nx  + mu * P
    U, _ = jsp.linalg.cho_factor(G)

    return K, k, P, p, U

@jax.jit
def lqr_regularized_backward(Qs, qs, Rs, rs, Ss, As, Bs, cs, mu):
    
    T = Qs.shape[0] - 1
    n = Qs.shape[1]
    m = Rs.shape[1]

    Ps = jnp.zeros((T + 1, n, n))
    ps = jnp.zeros((T + 1, n))
    Ks = jnp.zeros((T, m, n))
    ks = jnp.zeros((T, m))
    Us = jnp.zeros((T + 1, n, n))

    Ps = Ps.at[-1].set(Qs[T])
    ps = ps.at[-1].set(qs[T])

    G_T = jnp.eye(n) + mu * Qs[T]
    U_T, _ = jsp.linalg.cho_factor(G_T)
    Us = Us.at[T].set(U_T)
    
    def body(tt, inputs):
        Ks, ks, Ps, ps, Us = inputs         
        t = T - tt - 1
        K_t, k_t, P_t, p_t, U_t = \
            lqr_regularized_step(Qs[t], Rs[t], Ss[t], qs[t], rs[t], Ps[t+1], ps[t+1], As[t], Bs[t], cs[t], Us[t+1], mu)
        
        Ks = Ks.at[t].set(K_t)
        ks = ks.at[t].set(k_t)
        Ps = Ps.at[t].set(P_t)
        ps = ps.at[t].set(p_t)
        Us = Us.at[t].set(U_t)
        
        return Ks, ks, Ps, ps, Us
        
    return lax.fori_loop(0, T, body, (Ks, ks, Ps, ps, Us)) 

@jax.jit
def lqr_regularized_forward(x0, Ks, ks, As, Bs, cs, Us, ps, mu):
    T, m, n = Ks.shape
    xs = jnp.zeros((T + 1, n))
    us = jnp.zeros((T, m))
    xs = xs.at[0].set(x0)
    
    def body(t, inputs):
        xs, us = inputs
        u_t = Ks[t] @ xs[t] + ks[t]
        x_tmp = As[t] @ xs[t] + Bs[t] @ u_t + cs[t] - mu * ps[t+1]
        x_t1 = jsp.linalg.cho_solve((Us[t+1], False), x_tmp)
        us = us.at[t].set(u_t)
        xs = xs.at[t+1].set(x_t1)

        return xs, us
    
    return lax.fori_loop(0, T, body, (xs, us))