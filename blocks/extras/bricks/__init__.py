from theano import shared, tensor
from blocks.bricks import Feedforward, LinearActivation, Initializable
from blocks.bricks.base import application, lazy
from blocks.extras.initialization import PermutationMatrix
from blocks.extras.utils import check_valid_permutation
from blocks.initialization import Constant
from blocks.utils import shared_floatx, shared_floatx_nans
from blocks.roles import PARAMETER, add_role


class FixedPermutation(Feedforward):
    """Perform a fixed permutation of the input features.

    Parameters
    ----------
    order : ndarray-like
        A 1-dimensional container containing a permutation
        on the integers.
    dot : bool, optional
        Whether or not to perform the permutation by matrix
        multiplication. This may be faster in some circumstances
        but requires allocation of a permutation matrix.

    """
    @lazy(allocation=['order'])
    def __init__(self, order, dot=True, **kwargs):
        self.order = order
        self._dot = dot
        super(FixedPermutation, self).__init__(**kwargs)

    def _allocate(self):
        self.order = check_valid_permutation(self.order)
        if self.input_dim != len(self.order):
            raise ValueError("input_dim does not match length of order "
                             "vector")
        # No roles assigned here, since these are not learnable parameters.
        if self._dot:
            shape = (self.order.shape[0], self.order.shape[0])
            self._matrix = shared_floatx(
                PermutationMatrix(self.order).generate(None, shape))
        else:
            order = self.order.astype('int32')
            assert order.min() == 0  # Catch highly unlikely downcast issue.
            self._permutation = shared(order)

    @property
    def input_dim(self):
        return len(self.order)

    @application(inputs=['input_'], outputs=['output_'])
    def apply(self, input_):
        if self._dot:
            return tensor.dot(input_, self._matrix)
        else:
            return tensor.take(input_, self._permutation, axis=1)


class PReLU(Feedforward, Initializable):
    """Rectifier with a learned negative slope.

    Parameters
    ----------
    dim : int
        The dimension of the input. Required if slopes are not tied
        by :meth:`~.Brick.allocate`.
    tie_slopes : bool, default False
        Do all units use the same negative slope, or is the slope specific
        to each unit? When slopes are tied it is not necessary to specify
        the dim - it is not used.
    slopes_init : object, default Constant(0.25)
        A `NdarrayInitialization` instance which will be used by to
        initialize the slopes matrix. Required by
        :meth:`~.Brick.initialize`.

    See also: [He2015]_.

    .. [He2015] K. He at al. Delving Deep into Rectifiers:
        Surpassing Human-Level Performance on ImageNet Classification

    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, tie_slopes=False, slopes_init=None, **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.dim = dim
        self.tie_slopes = tie_slopes
        if self.tie_slopes and self.dim is None:
            self.dim = 1
        # This needs a fix with proper initialization selection
        # See also Blocks #740
        if slopes_init is None:
            slopes_init = Constant(0.25)
        self.slopes_init = slopes_init

    def _allocate(self):
        dim = self.output_dim
        if self.tie_slopes:
            dim = 1
        a = shared_floatx_nans((dim,), name='a',
                               broadcastable=(self.tie_slopes,))
        add_role(a, PARAMETER)
        self.parameters.append(a)

    def _initialize(self):
        a, = self.parameters
        self.slopes_init.initialize(a, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the PReLU activation: x if x>=0 else a*x."""
        a, = self.parameters
        return tensor.switch(input_ >= 0, input_, input_ * a)

    def get_dim(self, name):
        if name in ['input_', 'output']:
            return self.dim
        super(PReLU, self).get_dim(name)

    def _get_dim(self):
        return self.dim

    def _set_dim(self, value):
        self.dim = value

    input_dim = output_dim = property(_get_dim, _set_dim)


class LinearPReLU(LinearActivation):
    @lazy()
    def __init__(self, **kwargs):
        slopes_init = kwargs.pop('slopes_init', None)
        super(LinearPReLU, self).__init__(
            activation=PReLU(slopes_init=slopes_init),
            **kwargs)

    @property
    def output_dim(self):
        return self.linear.output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.linear.output_dim = value
        self.activation.output_dim = value
