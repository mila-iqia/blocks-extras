import numpy
import theano
from numpy.testing import assert_allclose
from theano import tensor

from blocks.bricks import Identity
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.attention import SequenceContentAttention
from blocks.initialization import IsotropicGaussian, Constant
from blocks_extras.bricks.attention2 import AttentionRecurrent
from blocks.graph import ComputationGraph
from blocks.select import Selector


def test_attention_recurrent():
    rng = numpy.random.RandomState(1234)

    dim = 5
    batch_size = 4
    input_length = 20

    attended_dim = 10
    attended_length = 15

    wrapped = SimpleRecurrent(dim, Identity())
    attention = SequenceContentAttention(
        state_names=wrapped.apply.states,
        attended_dim=attended_dim, match_dim=attended_dim)
    recurrent = AttentionRecurrent(wrapped, attention, seed=1234)
    recurrent.weights_init = IsotropicGaussian(0.5)
    recurrent.biases_init = Constant(0)
    recurrent.initialize()

    attended = tensor.tensor3("attended")
    attended_mask = tensor.matrix("attended_mask")
    inputs = tensor.tensor3("inputs")
    inputs_mask = tensor.matrix("inputs_mask")
    outputs = recurrent.apply(
        inputs=inputs, mask=inputs_mask,
        attended=attended, attended_mask=attended_mask)
    states, glimpses, weights = outputs
    assert states.ndim == 3
    assert glimpses.ndim == 3
    assert weights.ndim == 3

    # For values.
    def rand(size):
        return rng.uniform(size=size).astype(theano.config.floatX)

    # For masks.
    def generate_mask(length, batch_size):
        mask = numpy.ones((length, batch_size), dtype=theano.config.floatX)
        # To make it look like read data
        for i in range(batch_size):
            mask[1 + rng.randint(0, length - 1):, i] = 0.0
        return mask

    input_vals = rand((input_length, batch_size, dim))
    input_mask_vals = generate_mask(input_length, batch_size)
    attended_vals = rand((attended_length, batch_size, attended_dim))
    attended_mask_vals = generate_mask(attended_length, batch_size)

    func = theano.function([inputs, inputs_mask, attended, attended_mask],
                           [states, glimpses, weights])
    states_vals, glimpses_vals, weight_vals = func(
        input_vals, input_mask_vals,
        attended_vals, attended_mask_vals)
    assert states_vals.shape == (input_length, batch_size, dim)
    assert glimpses_vals.shape == (input_length, batch_size, attended_dim)

    assert (len(ComputationGraph(outputs).shared_variables) ==
            len(Selector(recurrent).get_parameters()))

    # Manual reimplementation
    inputs2d = tensor.matrix()
    states2d = tensor.matrix()
    mask1d = tensor.vector()
    weighted_averages = tensor.matrix()
    distribute_func = theano.function(
        [inputs2d, weighted_averages],
        recurrent.distribute.apply(
            inputs=inputs2d,
            weighted_averages=weighted_averages))
    wrapped_apply_func = theano.function(
        [states2d, inputs2d, mask1d], wrapped.apply(
            states=states2d, inputs=inputs2d, mask=mask1d, iterate=False))
    attention_func = theano.function(
        [states2d, attended, attended_mask],
        attention.take_glimpses(
            attended=attended, attended_mask=attended_mask,
            states=states2d))
    states_man = wrapped.initial_states(batch_size).eval()
    glimpses_man = numpy.zeros((batch_size, attended_dim),
                               dtype=theano.config.floatX)
    for i in range(input_length):
        inputs_man = distribute_func(input_vals[i], glimpses_man)
        states_man = wrapped_apply_func(states_man, inputs_man,
                                        input_mask_vals[i])
        glimpses_man, weights_man = attention_func(
            states_man, attended_vals, attended_mask_vals)
        assert_allclose(states_man, states_vals[i], rtol=1e-5)
        assert_allclose(glimpses_man, glimpses_vals[i], rtol=1e-5)
        assert_allclose(weights_man, weight_vals[i], rtol=1e-5)

    # weights for not masked position must be zero
    assert numpy.all(weight_vals * (1 - attended_mask_vals.T) == 0)
    # weights for masked positions must be non-zero
    assert numpy.all(abs(weight_vals + (1 - attended_mask_vals.T)) > 1e-5)
    # weights from different steps should be noticeably different
    assert (abs(weight_vals[0] - weight_vals[1])).sum() > 1e-2
    # weights for all state after the last masked position should be same
    for i in range(batch_size):
        last = int(input_mask_vals[:, i].sum())
        for j in range(last, input_length):
            assert_allclose(weight_vals[last, i], weight_vals[j, i], 1e-5)
