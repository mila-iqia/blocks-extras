import os
from collections import OrderedDict
from tempfile import gettempdir

import numpy

from theano import tensor

from fuel.schemes import SequentialScheme
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from blocks.utils.testing import MockMainLoop
from blocks.extensions import FinishAfter
from blocks_extras.extensions.predict import PredictDataStream


def test_predict():
    tempfile_path = os.path.join(gettempdir(), 'test_predict.npz')

    # set up mock datastream
    source = [[1], [2], [3], [4]]
    dataset = IndexableDataset(OrderedDict([('input', source)]))
    scheme = SequentialScheme(dataset.num_examples, batch_size=2)
    data_stream = DataStream(dataset, iteration_scheme=scheme)

    # simulate small "network" that increments the input by 1
    input_tensor = tensor.matrix('input')
    output_tensor = input_tensor + 1
    output_tensor.name = 'output_tensor'

    main_loop = MockMainLoop(extensions=[
        PredictDataStream(data_stream=data_stream,
                          variables=[output_tensor],
                          path=tempfile_path,
                          after_training=True),
        FinishAfter(after_n_epochs=1)
    ])
    main_loop.run()

    # assert resulting prediction is saved
    prediction = numpy.load(tempfile_path)
    assert numpy.all(prediction[output_tensor.name] == numpy.array(source) + 1)

    try:
        os.remove(tempfile_path)
    except:
        pass
