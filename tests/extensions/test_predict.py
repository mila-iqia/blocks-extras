import numpy
import os
import pickle
from collections import OrderedDict
from tempfile import gettempdir

from theano import tensor

from fuel.schemes import SequentialScheme
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from blocks.utils.testing import MockMainLoop
from blocks.extensions import FinishAfter
from blocks_extras.extensions.predict import PredictDataStream


def test_predict():
    tempfile = os.path.join(gettempdir(), 'test_predict.pkl')

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
                          path=tempfile,
                          after_training=True),
        FinishAfter(after_n_epochs=1)
    ])
    main_loop.run()

    # assert resulting prediction is saved
    prediction = pickle.load(open(tempfile, 'rb'))
    assert numpy.all(prediction[output_tensor.name] == numpy.array(source) + 1)

    try:
        os.remove(tempfile)
    except:
        pass
