"""Simple helpers for converting continuous-time dynamics to discrete-time."""

def euler(dynamics, dt=0.01):
  return lambda x, u, t: x + dt * dynamics(x, u, t)


def rk4(dynamics, dt=0.01):
  def integrator(x, u, t):
    dt2 = dt / 2.0
    k1 = dynamics(x, u, t)
    k2 = dynamics(x + dt2 * k1, u, t)
    k3 = dynamics(x + dt2 * k2, u, t)
    k4 = dynamics(x + dt * k3, u, t)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
  return integrator