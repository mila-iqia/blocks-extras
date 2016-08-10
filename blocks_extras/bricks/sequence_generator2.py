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
class AbstractReadout(Initializable):
    """The interface for the readout component of a sequence generator.

    Attributes
    ----------
    input_names : dict
    input_dims : dict

    """
    @lazy(allocation=['input_dims'])
    def __init__(self, input_names, input_dims, **kwargs):
        self.input_names = input_names
        self.input_dims = input_dims
        super(AbstractReadout, self).__init__(**kwargs)

    @abstractmethod
    def costs(self, prediction, prediction_mask,
              groundtruth, groundtruth_mask, **inputs):
        """Compute the costs.

        Can accept sequences and contexts to compute complicated costs
        such as e.g. REINFORCE with baseline.

        """
        pass

    @abstractmethod
    def scores(self, **inputs):
        """Compute the score for all possible next tokens.

        This is only needed for the beam search and can only be implement
        for discrete output tokens.

        """
        pass

    @abstractmethod
    def sample(self, **inputs):
        pass


@add_metaclass(ABCMeta)
class MergeReadout(AbstractReadout):
    """Readout that merges its inputs first."""
    @lazy(allocation=['merge_dim', 'post_merge_dim'])
    def __init__(self, merge_dim, post_merge_dim,
                 merge_names=None,  merge=None, merge_prototype=None,
                 post_merge=None, **kwargs):
        super(MergeReadout, self).__init__(**kwargs)

        if not merge_dim:
            merge_dim = post_merge_dim
        if not merge_names:
            merge_names = kwargs['input_names']
        if not merge:
            merge = Merge(input_names=merge_names,
                          prototype=merge_prototype)
        if not post_merge:
            post_merge = Bias(dim=post_merge_dim)

        self.merge_names = merge_names
        self.merge_dim = merge_dim
        self.merge_brick = merge
        self.post_merge = post_merge
        self.post_merge_dim = post_merge_dim

        self.children = [self.merge_brick, self.post_merge]

    def _push_allocation_config(self):
        self.merge_brick.input_dims = [
            self.get_dim(name) for name in self.merge_names]
        self.merge_brick.output_dim = self.merge_dim
        self.post_merge.input_dim = self.merge_dim
        self.post_merge.output_dim = self.post_merge_dim

    @abstractmethod
    def costs(self, prediction, prediction_mask,
              groundtruth, groundtruth_mask, **inputs):
        """Compute the costs.

        Can accept sequences and contexts to compute complicated costs
        such as e.g. REINFORCE with baseline.

        """
        pass

    @abstractmethod
    def scores(self, **inputs):
        """Compute the score for all possible next tokens.

        This is only needed for the beam search and can only be implement
        for discrete output tokens.

        """
        pass

    @abstractmethod
    def sample(self, **inputs):
        pass

    @application
    def merge(self, **inputs):
        merged = self.merge_brick.apply(**inputs)
        merged = self.post_merge.apply(merged)
        return merged

    def get_dim(self, name):
        try:
            return self.input_dims[self.input_names.index(name)]
        except ValueError:
            return super(MergeReadout, self).get_dim(name)


class SoftmaxReadout(MergeReadout, Random):

    def __init__(self, num_tokens, **kwargs):
        kwargs['post_merge_dim'] = num_tokens
        super(SoftmaxReadout, self).__init__(**kwargs)
        self.num_tokens = num_tokens
        self.softmax = NDimensionalSoftmax()
        self.children += [self.softmax]

        self.costs.inputs = [
            'prediction', 'prediction_mask',
            'groundtruth', 'groundtruth_mask']
        self.all_scores.inputs = ['prediction']
        self.scores.inputs = []
        self.sample.inputs = []
        for application_method in [self.costs, self.all_scores,
                                   self.scores, self.sample]:
            application_method.inputs += self.input_names

        self.sample.outputs = ['samples', 'scores']

    @application
    def costs(self, prediction, prediction_mask,
              groundtruth, groundtruth_mask, **inputs):
        log_probs = self.all_scores(
            prediction, self.merge(**dict_subset(inputs, self.merge_names)))
        if not prediction_mask:
            prediction_mask = 1
        return -(log_probs * prediction_mask).sum(axis=0)

    @application
    def all_scores(self, prediction, merged):
        return -self.softmax.categorical_cross_entropy(
            prediction, merged, extra_ndim=1)

    @application
    def scores(self, **inputs):
        return self.softmax.log_probabilities(self.merge(
            **dict_subset(inputs, self.merge_names)))

    @application
    def sample(self, **inputs):
        scores = self.scores(**inputs)
        probs = tensor.exp(scores)
        sample = self.theano_rng.multinomial(pvals=probs).argmax(axis=1)
        return sample, scores[tensor.arange(probs.shape[0]), sample]

    def get_dim(self, name):
        if name == 'samples' or name == 'scores':
            return 0
        return super(SoftmaxReadout, self).get_dim(name)


@add_metaclass(ABCMeta)
class Feedback(Initializable):
    """Feedback.

    Attributes
    ----------
    output_names : list
    output_dims : dict

    """
    @lazy(allocation=['output_names', 'output_dims'])
    def __init__(self, output_names, output_dims,
                 embedding=None, input_dim=0,
                 **kwargs):
        super(Feedback, self).__init__(**kwargs)

        self.output_names = output_names
        self.output_dims = output_dims
        self.input_dim = input_dim

        self.embedding = embedding
        self.fork = Fork(self.output_names)

        self.apply.inputs = ['input']
        self.apply.outputs = output_names

        self.children = [self.embedding, self.fork]
        self.children = [child for child in self.children if child]

    def _push_allocation_config(self):
        if self.fork:
            self.fork.output_dims = self.output_dims
        else:
            self.embedding.output_dim, = self.output_dims
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
        self.readout.input_dims = [
            self.recurrent.get_dim(name)
            for name in self.readout.input_names]
        self.feedback.output_dims = [
            self.recurrent.get_dim(name)
            for name in self.feedback.output_names]

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
        # These variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        for name in states_outputs:
            application_call.add_auxiliary_variable(
                states_outputs[name][-1].copy(), name=name+"_final_value")
        # Discard the final states
        for name in self.recurrent.apply.states:
            states_outputs[name] = states_outputs[name][:-1]
        # Add all states and outputs and auxiliary variables
        for name, variable in list(states_outputs.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)

        # Those can potentially be used for computing the cost.
        sequences_contexts = dict_subset(
            sequences_states_contexts,
            self.generate.contexts, self.generate.sequences)
        return self.readout.costs(
            prediction, prediction_mask,
            groundtruth, groundtruth_mask,
            **dict_subset(dict_union(states_outputs,
                                     sequences_contexts),
                          self.readout.costs.inputs,
                          must_have=False))

    @recurrent
    def generate(self, **sequences_states_contexts):
        sampling_inputs = dict_subset(
            sequences_states_contexts, self.readout.sample.inputs)
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
