import sys
import time
import logging

import numpy

from blocks.extensions import SimpleExtension
from platoon.channel import Worker, Controller

logger = logging.getLogger(__name__)

class Synchronize(SimpleExtension):
    def __init__(self, worker, **kwargs):
        kwargs.setdefault("before_training", True)
        kwargs.setdefault("after_training", True)
        super(Synchronize, self).__init__(**kwargs)
        self.worker = worker

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['worker']
        return state

    def do(self, which_callback, *args):
        if which_callback == 'before_training':
            self.worker.init_shared_params(self.main_loop.model.parameters)
            if not self.worker.is_main_worker:
                self.worker.copy_to_local()
                self.worker.wait_for_initialization()
                logger.debug("Copied parameters from shared")
            else:
                self.worker.copy_to_global()
                self.worker.report_initialization()
                logger.debug("Initialized shared parameters")
        elif which_callback == 'after_training':
            self.worker.send_req('done')
        else:
            self.worker.sync_params()


class SynchronizeWorker(Worker):

    def __init__(self, job_name, sync_rule, *args, **kwargs):
        self.job_name = job_name
        self.sync_rule = sync_rule
        super(SynchronizeWorker, self).__init__(*args, **kwargs)

    def init_shared_params(self, parameters):
        super(SynchronizeWorker, self).init_shared_params(
           self.job_name, parameters, self.sync_rule)

    @property
    def is_main_worker(self):
        if not hasattr(self, '_is_main_worker'):
            self._is_main_worker = self.send_req('is_main_worker?')
        return self._is_main_worker

    @property
    def seed(self):
        if not hasattr(self, '_seed'):
            self._seed = self.send_req('seed')
        return self._seed

    def report_initialization(self):
        if not self.is_main_worker:
            raise ValueError("Only main worker can report initialization")
        self.send_req('initialized')

    def wait_for_initialization(self):
        while not self.send_req('initialized?'):
            time.sleep(0.01)


class SynchronizeController(Controller):
    """Controls synchronization of several training jobs.

    This controller is necessary to make sure that:
    - one of the workers is chosen as the main worker
    - other workers start working only after main worker initializes all
      the shared parameters parameters
    - each worker receives a unique random seed, which is meant to determine
      the order of data traversal

    Parameters
    ----------
    seed_for_seeds : int
        The seed to be used in the random number generator that provides
        the seeds to the workers.

    """
    def __init__(self, seed_for_seeds=1):
        super(SynchronizeController, self).__init__()
        self.main_worker = None
        self.parameters_initialized = False
        self.seed_generator = numpy.random.RandomState(seed_for_seeds)

    def handle_control(self, req, worker_id):
        if req == 'is_main_worker?':
            print 'is_main_worker?', worker_id
            if not self.main_worker:
                self.main_worker = worker_id
            result = self.main_worker == worker_id
            print result
            return result
        elif req == 'initialized?':
            print 'initialized?'
            return self.parameters_initialized
        elif req == 'initialized':
            print 'initialized'
            self.parameters_initialized = True
        elif req == 'seed':
            seed = self.seed_generator.randint(1000)
            print 'seed', worker_id, seed
            return seed
        elif req == 'done':
            print 'done', worker_id
            self.worker_is_done(worker_id)
        else:
            raise ValueError("Unknown request " + req)


if __name__ == '__main__':
    import sys
    controller = SynchronizeController()
    controller.init_control(1111)
    controller.serve()
