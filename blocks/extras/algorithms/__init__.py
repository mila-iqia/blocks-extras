from blocks.algorithms import CompositeRule, Scale, StepRule
from blocks.utils import shared_floatx


class BasicNesterovMomentum(StepRule):
    """Accumulates step with exponential discount.

    Parameters
    ----------
    momentum : float, optional
        The momentum coefficient. Defaults to 0.

    Notes
    -----
    This step rule is intended to be used in conjunction with another
    step rule, _e.g._ :class:`Scale`. For an all-batteries-included
    experience, look at :class:`NesterovMomentum`.

    """
    def __init__(self, momentum=0.):
        self.momentum = shared_floatx(momentum)

    def compute_step(self, parameter, previous_step):
        velocity = shared_floatx(parameter.get_value() * 0.)
        velocity_update = self.momentum*velocity + previous_step
        step = (self.momentum**2 * velocity + previous_step *
                (1 + self.momentum))
        updates = [(velocity, velocity_update)]
        return step, updates


class NesterovMomentum(CompositeRule):
    """Accumulates step with exponential discount.
    Combines :class:`BasicNesterovMomentum` and :class:`Scale` to form the
    usual Nesterov momentum step rule.

    Parameters
    ----------
    learning_rate : float, optional
        The learning rate by which the previous step scaled. Defaults to 1.
    momentum : float, optional
        The momentum coefficient. Defaults to 0.

    Attributes
    ----------
    learning_rate : :class:`~tensor.SharedVariable`
        A variable for learning rate.
    momentum : :class:`~tensor.SharedVariable`
        A variable for momentum.

    See Also
    --------
    :class:`SharedVariableModifier`

    """
    def __init__(self, learning_rate=1.0, momentum=0.):
        scale = Scale(learning_rate=learning_rate)
        basic_nesterov_momentum = BasicNesterovMomentum(momentum=momentum)
        self.learning_rate = scale.learning_rate
        self.momentum = basic_nesterov_momentum.momentum
        self.components = [scale, basic_nesterov_momentum]
