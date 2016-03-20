from collections import OrderedDict

import numpy

from blocks.extensions import SimpleExtension
from blocks.graph import ComputationGraph


class PredictDataStream(SimpleExtension):
    """Evaluate tensors for a given datastream.

    Parameters
    ----------
    data_stream : `~fuel.streams.DataStream`
        Data stream for which to evaluate.
    variables : list of `~theano.TensorVariable`
        The variables to evaluate.
    path : str, optional
        The destination path for pickling. If not given, the output
        will not be saved, but will be stored in the `prediction`
        attribute instead.

    Attributes
    ----------
    prediction : `collections.OrderedDict`
        Storage for tensor evaluations. Maps tensor names to outputs
        of type `numpy.ndarray`. If the callback has not been invoked,
        this attribute is set to None.

    """
    def __init__(self, data_stream, variables, path=None, **kwargs):
        self.data_stream = data_stream
        self.variables = variables
        self.path = path
        self.prediction = None

        kwargs.setdefault('after_training', True)
        super(PredictDataStream, self).__init__(**kwargs)

        cg = ComputationGraph(variables)
        self.theano_function = cg.get_theano_function()

    def do(self, which_callback, *args):
        """Evaluate tensors for the given datastream."""
        predictions = OrderedDict([(var.name, []) for var in self.variables])
        for batch in self.data_stream.get_epoch_iterator(as_dict=True):
            prediction = self.theano_function(**batch)
            for var, pred in zip(self.variables, prediction):
                predictions[var.name].append(pred)

        # accumulate predictions for the entire epoch
        for var in self.variables:
            predictions[var.name] = numpy.concatenate(predictions[var.name],
                                                      axis=0)
        if self.path is not None:
            numpy.savez(self.path, **predictions)
