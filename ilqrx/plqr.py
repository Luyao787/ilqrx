import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax

@jax.jit
def lqr_par_backward_init_last(QT, qT):
    nx = QT.shape[0]
    P = QT
    p = -qT
    A = jnp.zeros((nx, nx))
    C = jnp.zeros((nx, nx))
    c = jnp.zeros((nx, 1))
    
    return P, p, A, C, c

@jax.jit
def lqr_par_backward_init_most(Qs, qs, Rs, rs, Ms, As, Bs, cs):    
    chofact = jsp.linalg.cho_factor(Rs)
    Mts = Ms.swapaxes(-2, -1)
    Bts = Bs.swapaxes(-2, -1)

    Ps_ = Qs - Mts @ jsp.linalg.cho_solve(chofact, Ms)
    ps_ = Mts @ jsp.linalg.cho_solve(chofact, rs) - qs
    As_ = As - Bs @ jsp.linalg.cho_solve(chofact, Ms)
    Cs_ = Bs @ jsp.linalg.cho_solve(chofact, Bts)
    cs_ = cs - Bs @ jsp.linalg.cho_solve(chofact, rs)

    return Ps_, ps_, As_, Cs_, cs_

@jax.jit
def _lqr_par_comb_V(P_ij, p_ij, A_ij, C_ij, c_ij,
                    P_jk, p_jk, A_jk, C_jk, c_jk):
    I = jnp.expand_dims(jnp.eye(A_ij.shape[-2]), 0)
    lufac = jsp.linalg.lu_factor(I + C_ij @ P_jk)
    D = jsp.linalg.lu_solve(lufac, A_ij) # D = (I + C_ij P_jk)^{-1} A_ij
    Dt = D.swapaxes(-2, -1)
    At_jk = A_jk.swapaxes(-2, -1)
    E = jsp.linalg.lu_solve(lufac, At_jk, trans=1) # E = (I + P_jk C_ij)^{-1} A_jk^T
    Et = E.swapaxes(-2, -1)

    A_ik = A_jk @ jsp.linalg.lu_solve(lufac, A_ij)
    C_ik = Et @ C_ij @ At_jk + C_jk
    P_ik = Dt @ P_jk @ A_ij + P_ij
    c_ik = Et @ (c_ij + C_ij @ p_jk) + c_jk
    p_ik = Dt @ (p_jk - P_jk @ c_ij) + p_ij

    return P_ik, p_ik, A_ik, C_ik, c_ik    

@jax.jit
def lqr_par_comb_V(elem_ij, elem_jk):
    
    P_ij, p_ij, A_ij, C_ij, c_ij = elem_ij
    P_jk, p_jk, A_jk, C_jk, c_jk = elem_jk
     
    P_ik, p_ik, A_ik, C_ik, c_ik = _lqr_par_comb_V(P_ij, p_ij, A_ij, C_ij, c_ij,
                                                   P_jk, p_jk, A_jk, C_jk, c_jk)    
     
    return P_ik, p_ik, A_ik, C_ik, c_ik

@jax.jit
def lqr_par_comb_V_reverse(elem_jk, elem_ij):
    return lqr_par_comb_V(elem_ij, elem_jk)

@jax.jit
def lqr_par_backward(Qs, qs, Rs, rs, Ms, As, Bs, cs):
    qs = jnp.expand_dims(qs, axis=-1)
    rs = jnp.expand_dims(rs, axis=-1)
    cs = jnp.expand_dims(cs, axis=-1)
    elems_most = lqr_par_backward_init_most(Qs[:-1], qs[:-1], Rs, rs, Ms, As, Bs, cs)    
    elems_last = lqr_par_backward_init_last(Qs[-1], qs[-1])
    
    elems = tuple(jnp.concatenate([em, jnp.expand_dims(el, 0)], axis=0) 
                  for em, el in zip(elems_most, elems_last))
    elems = lax.associative_scan(lqr_par_comb_V_reverse, elems, reverse=True)
        
    Ps = elems[0]
    ps = elems[1]
        
    # return Ps, -ps.squeeze(axis=-1)
    
    Bts = jnp.swapaxes(Bs, -2, -1)
    Quus = Rs + jnp.matmul(Bts, jnp.matmul(Ps[1:, ...], Bs))
    chofac = jsp.linalg.cho_factor(Quus)
      
    Hs = jnp.matmul(Bts, jnp.matmul(Ps[1:, ...], As)) + Ms
    hs = jnp.matmul(Bts, (jnp.matmul(Ps[1:, ...], cs) - ps[1:, ...])) + rs
    Ks = -jsp.linalg.cho_solve(chofac, Hs)    
    ks = -jsp.linalg.cho_solve(chofac, hs).squeeze(axis=-1)

    ps = -ps.squeeze(axis=-1)
        
    return Ks, ks, Ps, ps


@jax.jit
def lqr_par_forward_init_first(x0, K, k, A, B, c):
    tA =  jnp.zeros_like(A)
    tc = jnp.matmul(A + jnp.matmul(B, K), x0) + jnp.matmul(B, k) + c
    return tA, tc

@jax.jit
def lqr_par_forward_init_most(Ks, ks, As, Bs, cs):
    tA = As + jnp.matmul(Bs, Ks)
    tc = cs + jnp.matmul(Bs, ks)
    return tA, tc

@jax.jit
def lqr_par_forward_comb(elemij, elemjk):
    Aij, cij = elemij
    Ajk, cjk = elemjk
    Aik = jnp.matmul(Ajk, Aij)
    cik = jnp.matmul(Ajk, cij) + cjk
    return Aik, cik
    
@jax.jit
def lqr_par_forward(x0, Ks, ks, As, Bs, cs):
    # for batch computation
    ks = jnp.expand_dims(ks, axis=-1)
    cs = jnp.expand_dims(cs, axis=-1)
    x0 = jnp.expand_dims(x0, axis=-1)
    elems_first = lqr_par_forward_init_first(x0, Ks[0], ks[0], As[0], Bs[0], cs[0])
    elems_most = lqr_par_forward_init_most(Ks[1:], ks[1:], As[1:], Bs[1:], cs[1:])
    
    elems = tuple(jnp.concatenate([jnp.expand_dims(ef, 0), em], axis=0) for ef, em in zip(elems_first, elems_most))
    elems = lax.associative_scan(lqr_par_forward_comb, elems)
    xs = jnp.concatenate([jnp.expand_dims(x0, axis=0), elems[1]], axis=0)
    us = jnp.matmul(Ks, xs[:-1]) + ks
    
    return jnp.squeeze(xs, axis=-1), jnp.squeeze(us, axis=-1)