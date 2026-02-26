import abc
import jax

class AbstractConstraint(abc.ABC):
    @abc.abstractmethod
    def constraint_fn(self, x, u, t, *args):
        raise NotImplementedError

    def jacobian(self, x, u, t, *args):
        jac_x = jax.jacobian(self.constraint_fn)(x, u, t, *args)
        jac_u = jax.jacobian(self.constraint_fn, argnums=1)(x, u, t, *args)
        return jac_x, jac_u
