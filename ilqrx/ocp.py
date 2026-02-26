from typing import Callable
from ilqrx.abstract_dynamics import AbstractDynamics
from ilqrx.abstract_cost import AbstractCost
from ilqrx.abstract_constraint import AbstractConstraint

class OptimalControlProblem:
    def __init__(self, 
                 dynamic_model: AbstractDynamics, 
                 cost_model: AbstractCost,
                 equality_constraint_model: AbstractConstraint,
                 inequality_constraint_model: AbstractConstraint):
        self._dynamic_model = dynamic_model
        self._cost_model = cost_model
        self._equality_constraint_model = equality_constraint_model
        self._inequality_constraint_model = inequality_constraint_model

    @property
    def dynamic_model(self) -> AbstractDynamics:
        return self._dynamic_model

    @property
    def cost_model(self) -> AbstractCost:
        return self._cost_model

    @property
    def equality_constraint_model(self) -> AbstractConstraint:
        return self._equality_constraint_model
    
    @property
    def inequality_constraint_model(self) -> AbstractConstraint:
        return self._inequality_constraint_model