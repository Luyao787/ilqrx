import abc
import jax

class AbstractCost(abc.ABC):
    @abc.abstractmethod
    def cost_fn(self, x, u, t, *args):
        raise NotImplementedError

    def gradient(self, x, u, t, *args):
        grad_x = jax.grad(self.cost_fn)(x, u, t, *args)
        grad_u = jax.grad(self.cost_fn, argnums=1)(x, u, t, *args)
        return grad_x, grad_u
    
    def hessian(self, x, u, t, *args):
        hess_x = jax.hessian(self.cost_fn)(x, u, t, *args)
        hess_u = jax.hessian(self.cost_fn, argnums=1)(x, u, t, *args)
        hess_xu = jax.jacobian(jax.grad(self.cost_fn), argnums=1)(x, u, t, *args)
        return hess_x, hess_u, hess_xu