import numpy
from numpy.testing import assert_equal, assert_allclose
import theano
from theano import tensor

from blocks.initialization import Uniform
from blocks_extras.bricks.sequence_generator2 import SoftmaxReadout


class TestReadouts(object):

    def setUp(self):
        self.readout = SoftmaxReadout(
            input_names=['states1', 'states2'],
            num_tokens=4, input_dims=[2, 3],
            weights_init=Uniform(width=1.0),
            biases_init=Uniform(width=1.0),
            seed=1)
        self.readout.initialize()

        self.states1 = numpy.array(
            [[[1., 2.]], [[2., 1.]]],
            dtype=theano.config.floatX)
        self.states2 = numpy.array(
            [[[3., 4., 5.]], [[5., 4., 3.]]],
            dtype=theano.config.floatX)
        self.merged = (
            self.states1.dot(self.readout.merge.children[0].W.get_value()) +
            self.states2.dot(self.readout.merge.children[1].W.get_value()) +
            self.readout.post_merge.parameters[0].get_value())

    def test_merge(self):
        assert self.readout.merge_dim == self.readout.num_tokens
        merged = self.readout._merge(
            states1=self.states1, states2=self.states2).eval()
        assert_equal(merged, self.merged)

    def test_all_scores(self):
        assert (self.readout.all_scores.inputs ==
                ['prediction', 'states1', 'states2'])
        all_scores = self.merged - numpy.log(
            numpy.exp(self.merged).sum(axis=2, keepdims=True))
        assert_allclose(
            self.readout.all_scores(
                tensor.as_tensor_variable([[1], [2]]),
                states1=self.states1,
                states2=self.states2).eval(),
            [[all_scores[0][0][1]], [all_scores[1][0][2]]],
            rtol=1e-4)

    def test_scores(self):
        assert self.readout.scores.inputs == ['states1', 'states2']
        assert_allclose(
            self.readout.scores(states1=self.states1[0],
                                states2=self.states2[0]).eval(),
            self.merged[0] - numpy.log(numpy.exp(self.merged[0]).sum()),
            rtol=1e-4)

    def test_sample(self):
        assert self.readout.sample.inputs == ['states1', 'states2']
        # TODO: check the returned value
        self.readout.sample(states1=self.states1[0],
                            states2=self.states2[0])
