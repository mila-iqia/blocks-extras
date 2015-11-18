import numpy
from numpy.testing import assert_equal
from numpy.testing.utils import assert_allclose
import theano
from theano import tensor
from blocks.extras.bricks import FixedPermutation, PReLU, LinearPReLU
from blocks.bricks import LinearTanh, LinearRectifier
from blocks.initialization import Constant, IsotropicGaussian


def test_fixed_permutation():
    x = tensor.matrix()
    x_val = numpy.arange(15, dtype=theano.config.floatX).reshape((5, 3))
    perm = FixedPermutation([2, 0, 1])
    y = perm.apply(x)
    y_val = y.eval({x: x_val})
    assert_equal(x_val[:, [2, 0, 1]], y_val)
    perm = FixedPermutation([2, 0, 1], dot=False)
    y = perm.apply(x)
    assert_equal(x_val[:, [2, 0, 1]], y_val)


def test_prelu():
    x = tensor.matrix()

    prelu = PReLU(dim=3, tie_slopes=False)
    prelu.initialize()
    assert_equal(prelu.parameters[0].get_value().shape[0], 3)
    prelu.parameters[0].set_value(numpy.array([0, 1, -1],
                                              dtype=theano.config.floatX))
    y = prelu.apply(x)
    x_val = numpy.array([[10, -1, -2],
                         [-1, 1, 2]],
                        dtype=theano.config.floatX)
    assert_allclose(y.eval({x: x_val}),
                    [[10, -1, 2],
                     [0, 1, 2]])

    prelu = PReLU(dim=3, tie_slopes=True)
    prelu.initialize()
    assert_equal(prelu.parameters[0].get_value().shape[0], 1)
    prelu.parameters[0].set_value(numpy.array([-1],
                                              dtype=theano.config.floatX))
    y = prelu.apply(x)
    assert_allclose(y.eval({x: x_val}),
                    numpy.abs(x_val))


def test_linear_activations_with_prelu():
    x = tensor.matrix()
    linear_tanh = LinearTanh(weights_init=Constant(2),
                             biases_init=Constant(1))
    linear_tanh.input_dim = 16
    linear_tanh.output_dim = 8

    linear_tanh.initialize()
    y = linear_tanh.apply(x)
    x_val = numpy.ones((4, 16), dtype=theano.config.floatX)
    assert_allclose(
        y.eval({x: x_val}),
        (numpy.tanh(x_val.dot(2 * numpy.ones((16, 8))) +
                    numpy.ones((4, 8))).reshape(4, 8)))

    linear_rect = LinearRectifier(weights_init=IsotropicGaussian(),
                                  biases_init=Constant(0.1))
    linear_prelu = LinearPReLU(weights_init=IsotropicGaussian(),
                               biases_init=Constant(0.1),
                               slopes_init=Constant(0.)
                               )
    linear_rect.input_dim = 16
    linear_rect.output_dim = 8
    linear_prelu.input_dim = 16
    linear_prelu.output_dim = 8

    linear_rect.initialize()
    linear_prelu.initialize()
    for p1, p2 in zip(linear_prelu.linear.parameters,
                      linear_rect.linear.parameters):
        p1.set_value(p2.get_value())

    y_rect = linear_rect.apply(x)
    y_prelu = linear_prelu.apply(x)
    assert_allclose(
        y_rect.eval({x: x_val}),
        y_prelu.eval({x: x_val}))
