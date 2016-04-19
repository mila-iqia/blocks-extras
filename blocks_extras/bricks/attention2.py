from blocks.bricks import (
    application, Initializable)
from blocks.bricks.recurrent import recurrent
from blocks.bricks.parallel import Distribute
from blocks.utils import dict_subset, dict_union, pack


class AttentionRecurrent(Initializable):
    """Combines an attention mechanism and a recurrent transition.

    This brick equips a recurrent transition with an attention mechanism.
    In order to do this two more contexts are added: one to be attended and
    a mask for it. It is also possible to use the contexts of the given
    recurrent transition for these purposes and not add any new ones,
    see `add_context` parameter.

    At the beginning of each step the glimpses from the previous step
    are used to do the transition. After that, new glimpses are computed
    based on the new state of the recurrent network and using the glimpses
    from the previous step.

    To let the user control the way glimpses are used, this brick also
    takes a "distribute" brick as parameter that distributes the
    information from glimpses across the sequential inputs of the wrapped
    recurrent transition.

    Parameters
    ----------
    transition : :class:`.BaseRecurrent`
        The recurrent transition.
    attention : :class:`.Brick`
        The attention mechanism.
    distribute : :class:`.Brick`, optional
        Distributes the information from glimpses across the input
        sequences of the transition. By default a :class:`.Distribute` is
        used, and those inputs containing the "mask" substring in their
        name are not affected.
    add_contexts : bool, optional
        If ``True``, new contexts for the attended and the attended mask
        are added to this transition, otherwise existing contexts of the
        wrapped transition are used. ``True`` by default.
    attended_name : str
        The name of the attended context. If ``None``, "attended"
        or the first context of the recurrent transition is used
        depending on the value of `add_contents` flag.
    attended_mask_name : str
        The name of the mask for the attended context. If ``None``,
        "attended_mask" or the second context of the recurrent transition
        is used depending on the value of `add_contents` flag.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    Those coming to Blocks from Groundhog might recognize that this is
    a `RecurrentLayerWithSearch`, but on steroids :)

    """
    def __init__(self, transition, attention, distribute=None,
                 add_contexts=True,
                 attended_name=None, attended_mask_name=None,
                 **kwargs):
        super(AttentionRecurrent, self).__init__(**kwargs)
        self._sequence_names = list(transition.apply.sequences)
        self._state_names = list(transition.apply.states)
        self._context_names = list(transition.apply.contexts)
        if add_contexts:
            if not attended_name:
                attended_name = 'attended'
            if not attended_mask_name:
                attended_mask_name = 'attended_mask'
            self._context_names += [attended_name, attended_mask_name]
        else:
            attended_name = self._context_names[0]
            attended_mask_name = self._context_names[1]
        if not distribute:
            normal_inputs = [name for name in self._sequence_names
                             if 'mask' not in name]
            distribute = Distribute(normal_inputs,
                                    attention.take_glimpses.outputs[0])

        self.transition = transition
        self.attention = attention
        self.distribute = distribute
        self.add_contexts = add_contexts
        self.attended_name = attended_name
        self.attended_mask_name = attended_mask_name

        self.preprocessed_attended_name = "preprocessed_" + self.attended_name

        self._glimpse_names = self.attention.take_glimpses.outputs
        # We need to determine which glimpses are fed back.
        # Currently we extract it from `take_glimpses` signature.
        self.previous_glimpses_needed = [
            name for name in self._glimpse_names
            if name in self.attention.take_glimpses.inputs]

        self.do_apply.sequences = self._sequence_names
        self.do_apply.contexts = (self._context_names +
                                  [self.preprocessed_attended_name])
        self.do_apply.states = self._state_names + self._glimpse_names
        self.do_apply.outputs = self._state_names + self._glimpse_names
        self.initial_states.outputs = self.do_apply.states

        self.children = [self.transition, self.attention, self.distribute]

    def _push_allocation_config(self):
        self.attention.state_dims = self.transition.get_dims(
            self.attention.state_names)
        self.attention.attended_dim = self.get_dim(self.attended_name)
        self.distribute.source_dim = self.attention.get_dim(
            self.distribute.source_name)
        self.distribute.target_dims = self.transition.get_dims(
            self.distribute.target_names)

    @recurrent
    def do_apply(self, **kwargs):
        r"""Process a sequence attending the attended context every step.

        In addition to the original sequence this method also requires
        its preprocessed version, the one computed by the `preprocess`
        method of the attention mechanism. Unknown keyword arguments
        are passed to the wrapped transition.

        Parameters
        ----------
        \*\*kwargs
            Should contain current inputs, previous step states, contexts,
            the preprocessed attended context, previous step glimpses.

        Returns
        -------
        outputs : list of :class:`~tensor.TensorVariable`
            The current step states and glimpses.

        """
        attended = kwargs[self.attended_name]
        preprocessed_attended = kwargs.pop(self.preprocessed_attended_name)
        attended_mask = kwargs.get(self.attended_mask_name)
        if self.add_contexts:
            kwargs.pop(self.attended_name)
            kwargs.pop(self.attended_mask_name, None)
        sequences = dict_subset(kwargs, self._sequence_names, pop=True,
                                must_have=False)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        # By this time **kwargs will contain the states and the contexts
        # of the transition

        # Compute next states
        sequences_without_mask = {
            name: variable for name, variable in sequences.items()
            if 'mask' not in name}
        sequences.update(self.distribute.apply(
            as_dict=True, **dict_subset(
                dict_union(sequences_without_mask, glimpses),
                self.distribute.apply.inputs)))
        current_states = self.transition.apply(
            iterate=False, as_dict=True,
            **dict_union(sequences, kwargs))

        glimpses_needed = dict_subset(glimpses, self.previous_glimpses_needed)
        current_glimpses = self.attention.take_glimpses(
            as_dict=True,
            **dict_union(
                current_states, glimpses_needed,
                {self.attended_name: attended,
                 self.attended_mask_name: attended_mask,
                 self.preprocessed_attended_name: preprocessed_attended}))
        return list(current_states.values()) + list(current_glimpses.values())

    @application
    def apply(self, **kwargs):
        """Preprocess a sequence attending the attended context at every step.

        Preprocesses the attended context and runs :meth:`do_apply`. See
        :meth:`do_apply` documentation for further information.

        """
        preprocessed_attended = self.attention.preprocess(
            kwargs[self.attended_name])
        return self.do_apply(
            **dict_union(kwargs,
                         {self.preprocessed_attended_name:
                          preprocessed_attended}))

    @apply.delegate
    def apply_delegate(self):
        # TODO: Nice interface for this trick?
        return self.do_apply.__get__(self, None)

    @apply.property('contexts')
    def apply_contexts(self):
        return self._context_names

    @application
    def initial_states(self, batch_size, **kwargs):
        return (pack(self.transition.initial_states(
                     batch_size, **kwargs)) +
                pack(self.attention.initial_glimpses(
                     batch_size, kwargs[self.attended_name])))

    def get_dim(self, name):
        if name in self._glimpse_names:
            return self.attention.get_dim(name)
        if name == self.preprocessed_attended_name:
            (original_name,) = self.attention.preprocess.outputs
            return self.attention.get_dim(original_name)
        if self.add_contexts:
            if name == self.attended_name:
                return self.attention.get_dim(
                    self.attention.take_glimpses.inputs[0])
            if name == self.attended_mask_name:
                return 0
        return self.transition.get_dim(name)
