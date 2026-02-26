import jax
import jax.numpy as jnp

def rotation_matrix_from_quat(q):
    """Computes the rotation matrix from a unit quaternion."""
    qw, qx, qy, qz = q
    qw2 = qw * qw
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    return jnp.array(
        [
            [qw2 + qx2 - qy2 - qz2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), qw2 - qx2 + qy2 - qz2, 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw2 - qx2 - qy2 + qz2],
        ]
    )

def skew(v):
    """Returns the skew-symmetric matrix of a vector."""
    return jnp.array([[0., -v[2], v[1]],
                      [v[2], 0., -v[0]],
                      [-v[1], v[0], 0.]])

def left_matrix(q):
    """Returns the left quaternion multiplication matrix."""
    qw, qx, qy, qz = q
    return jnp.array([[qw, -qx, -qy, -qz],
                      [qx,  qw, -qz,  qy],
                      [qy,  qz,  qw, -qx],
                      [qz, -qy,  qx,  qw]])

def attitude_jacobian(q):
    """Computes the attitude Jacobian for a quaternion."""
    qw, qx, qy, qz = q
    return jnp.array([[-qx, -qy, -qz],
                      [ qw, -qz,  qy],
                      [ qz,  qw, -qx],
                      [-qy,  qx,  qw]])
    
def cayley_map(phi):
    """Computes the Cayley map from 3D vector to unit quaternion."""
    q = jnp.zeros(4)
    q = q.at[0].set(1.0)  # qw
    q = q.at[1:].set(phi)
    q /= jnp.sqrt(1 + jnp.dot(phi, phi))
    return q

def inv_cayley_map(q):
    """Computes the inverse Cayley map from unit quaternion to 3D vector."""
    phi = q[1:] / (1e-15 + q[0])
    return phi

def quat_conjugate(q):
    """Returns the conjugate of a quaternion."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

def quat_multiply(q1, q2):
    """Multiplies two quaternions (q1 * q2)."""
    L = left_matrix(q1)
    return L @ q2

def quat_slerp(q1, q2, alpha):
    """Spherical linear interpolation between two quaternions."""
    dot_product = jnp.dot(q1, q2)
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product 
    dot_product = jnp.clip(dot_product, -1.0, 1.0)
    theta = jnp.arccos(dot_product)
    sin_theta = jnp.sin(theta)
    if sin_theta < 1e-6:
        return (1.0 - alpha) * q1 + alpha * q2
    s0 = jnp.sin((1 - alpha) * theta) / sin_theta
    s1 = jnp.sin(alpha * theta) / sin_theta
    return s0 * q1 + s1 * q2

