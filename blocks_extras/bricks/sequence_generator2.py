"""Generating sequences with recurrent networks.

This module provides bricks that make it simple to generate sequences
with recurrent networks. Example of things that you can do with it
include language modelling, speech synthesis, machine translation.

Making a sequence generator from a recurrent brick amounts
to choosing a *readout* brick and a *feedback* brick. The readout brick
defines

- a differentiable *cost* associated with outputting a *prediction*, when
the corrent sequence was *groundtruth*

- how the network outputs *scores* for the next symbol given the
states of the recurrent brick

- (optionally) how the next symbol can be sampled given the states

The feedback brick defines how inputs of recurrent brick are computed
from output symbols.

Please note, that elements of the output sequence can also be continuous
values or vectors, provided that suitable readout and feedback are chosen.

"""
from abc import ABCMeta, abstractmethod
from six import add_metaclass

from blocks.bricks.base import lazy, application
from blocks.bricks.recurrent import recurrent
from blocks.bricks.simple import (
    Bias, Initializable, Random, NDimensionalSoftmax)
from blocks.bricks.parallel import Merge, Fork
from blocks.utils import dict_subset, dict_union

from theano import tensor


@add_metaclass(ABCMeta)
class Readout(Initializable):
    """Readout.

    """
    @lazy(allocation=['dim', 'state_dims', 'merged_dim', 'merged_states'])
    def __init__(self, dim, state_dims, merged_dim, merged_states,
                 merge=None, merge_prototype=None,
                 post_merge=None, **kwargs):
        super(Readout, self).__init__(**kwargs)

        if not merge:
            merge = Merge(input_names=merged_states,
                          prototype=merge_prototype)
        if not post_merge:
            post_merge = Bias(dim=dim)
        if not merged_dim:
            merged_dim = dim

        self.dim = dim
        self.state_dims = state_dims
        self.merged_dim = merged_dim
        self.merged_states = merged_states
        self.merge = merge
        self.post_merge = post_merge

        self.children = [self.merge, self.post_merge]

    def _push_allocation_config(self):
        self.merge.input_names = self.merged_states
        self.merge.input_dims = self.state_dims
        self.merge.output_dim = self.merged_dim
        self.post_merge.input_dim = self.merged_dim
        self.post_merge.output_dim = self.dim

    @abstractmethod
    def costs(self, prediction, prediction_mask,
              groundtruth, groundtruth_mask, **states):
        pass

    @abstractmethod
    def all_scores(self, prediction, **states):
        pass

    @abstractmethod
    def scores(self, **states):
        pass

    @abstractmethod
    def sample(self, **states):
        pass

    @application
    def _merge(self, **states):
        merged = self.merge.apply(**{name: states[name]
                                     for name in self.merge.input_names})
        merged = self.post_merge.apply(merged)
        return merged


class SoftmaxReadout(Readout, Random):

    def __init__(self, **kwargs):
        super(SoftmaxReadout, self).__init__(**kwargs)

        self.softmax = NDimensionalSoftmax()
        self.children += [self.softmax]

        self.costs.inputs = [
            'prediction', 'prediction_mask',
            'groundtruth', 'groundtruth_mask']
        self.all_scores.inputs = ['prediction']
        self.scores.inputs = self.sample.inputs = []
        for application_method in [self.costs, self.all_scores,
                                   self.scores, self.sample]:
            application_method.inputs += self.merged_states

        self.sample.outputs = ['samples', 'scores']

    @application
    def costs(self, prediction, prediction_mask,
              groundtruth, groundtruth_mask, **all_states):
        log_probs = self.all_scores(prediction, **all_states)
        if not prediction_mask:
            prediction_mask = 1
        return -(log_probs * prediction_mask).sum(axis=0)

    @application
    def all_scores(self, prediction, **all_states):
        return -self.softmax.categorical_cross_entropy(
            prediction, self._merge(**all_states), extra_ndim=1)

    @application
    def scores(self, **states):
        return self.softmax.log_probabilities(self._merge(**states))

    @application
    def sample(self, **states):
        scores = self.scores(**states)
        probs = tensor.exp(scores)
        sample = self.theano_rng.multinomial(pvals=probs).argmax(axis=1)
        return sample, scores[tensor.arange(probs.shape[0]), sample]

    def get_dim(self, name):
        if name == 'samples' or 'scores':
            return 0
        return super(SoftmaxReadout, self).get_dim(name)


@add_metaclass(ABCMeta)
class Feedback(Initializable):
    """Feedback.

    Attributes
    ----------
    feedback_sequences : list
    sequence_dims : dict

    """
    @lazy(allocation=['feedback_sequences', 'sequence_dims'])
    def __init__(self, feedback_sequences, sequence_dims,
                 embedding=None, input_dim=0,
                 **kwargs):
        super(Feedback, self).__init__(**kwargs)

        self.feedback_sequences = feedback_sequences
        self.sequence_dims = sequence_dims
        self.input_dim = input_dim

        self.embedding = embedding
        self.fork = Fork(self.feedback_sequences)

        self.apply.inputs = ['input']
        self.apply.outputs = feedback_sequences

        self.children = [self.embedding, self.fork]
        self.children = [child for child in self.children if child]

    def _push_allocation_config(self):
        if self.fork:
            self.fork.output_dims = self.sequence_dims
        else:
            self.embedding.output_dim, = self.sequence_dims
        if self.embedding:
            self.embedding.input_dim = self.input_dim
            self.fork.input_dim = self.embedding.output_dim
        else:
            self.fork.input_dim = self.input_dim

    @application
    def apply(self, symbols):
        embedded_symbols = symbols
        if self.embedding:
            embedded_symbols = self.embedding.apply(symbols)
        if self.fork:
            return self.fork.apply(embedded_symbols)
        return embedded_symbols


class SequenceGenerator(Initializable):

    def __init__(self, recurrent,  readout, feedback, **kwargs):
        super(SequenceGenerator, self).__init__(**kwargs)
        self.recurrent = recurrent
        self.readout = readout
        self.feedback = feedback
        self.children = [recurrent, readout, feedback]

        self.generate.sequences = self.recurrent.apply.sequences
        self.generate.states = self.recurrent.apply.states
        self.generate.contexts = self.recurrent.apply.contexts
        # TODO: allow recurrent to have outputs
        self.generate.outputs = (['samples', 'scores'] +
                                 self.recurrent.apply.outputs)
        self.initial_states.outputs = self.recurrent.initial_states.outputs

    def _push_allocation_config(self):
        self.readout.state_dims = [
            self.recurrent.get_dim(name)
            for name in self.readout.merged_states]
        self.feedback.sequence_dims = [
            self.recurrent.get_dim(name)
            for name in self.feedback.feedback_sequences]

    @application
    def costs(self, application_call,
              prediction, prediction_mask=None,
              groundtruth=None, groundtruth_mask=None,
              **sequences_states_contexts):
        feedback = self.feedback.apply(prediction, as_dict=True)
        states_outputs = self.recurrent.apply(
            mask=prediction_mask, return_initial_states=True, as_dict=True,
            # Using dict_union gives us a free sanity check that
            # the feedback entries do not override the ones
            # from sequences_states_contexts
            **dict_union(feedback, sequences_states_contexts))
        states_outputs = {name: states_outputs[name][:-1]
                          for name in states_outputs}

        for name, variable in list(states_outputs.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)
        # These variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        for name in states_outputs:
            application_call.add_auxiliary_variable(
                states_outputs[name][-1].copy(), name=name+"_final_value")

        return self.readout.costs(
            prediction, prediction_mask,
            groundtruth, groundtruth_mask,
            **dict_subset(states_outputs, self.readout.costs.inputs,
                          must_have=False))

    @recurrent
    def generate(self, **sequences_states_contexts):
        sampling_inputs = dict_subset(
            sequences_states_contexts, self.readout.sample.inputs,
            must_have=False)
        samples, scores = self.readout.sample(**sampling_inputs)
        feedback = self.feedback.apply(samples, as_dict=True)
        next_states_outputs = self.recurrent.apply(
            as_list=True, iterate=False,
            **dict_union(feedback, **sequences_states_contexts))
        return [samples, scores] + next_states_outputs

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        return self.recurrent.initial_states(batch_size,
                                             *args, **kwargs)

    def get_dim(self, name):
        if name == 'samples' or name == 'scores':
            return self.readout.get_dim(name)
        return self.recurrent.get_dim(name)

