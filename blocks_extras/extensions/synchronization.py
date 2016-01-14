import sys
import logging

import numpy

from blocks.extensions import SimpleExtension
from platoon.channel import Worker, Controller

logger = logging.getLogger(__name__)

class Synchronize(SimpleExtension):
    def __init__(self, worker, job_name, sync_rule, **kwargs):
        kwargs.setdefault("before_training", True)
        kwargs.setdefault("after_training", True)
        super(Synchronize, self).__init__(**kwargs)

        self.job_name = job_name
        self.sync_rule = sync_rule
        self.worker = worker

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['worker']
        return state

    def do(self, which_callback, *args):
        if which_callback == 'before_training':
            self.worker.init_shared_params(
                self.job_name, self.main_loop.model.parameters,
                self.sync_rule, cleanup=self.worker.is_main_worker)
            if not self.worker.is_main_worker:
                self.worker.copy_to_local()
                logger.debug("Copied parameters from shared")
            else:
                logger.debug("Initialized shared parameters")
        elif (which_callback == 'after_batch' or
              which_callback == 'after_epoch'):
            self.worker.sync_params(synchronous=True)
        elif which_callback == 'after_training':
            self.worker.send_req('done')


class SynchronizeWorker(Worker):

    def __init__(self, *args, **kwargs):
        super(SynchronizeWorker, self).__init__(*args, **kwargs)

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


class SynchronizeController(Controller):

    def __init__(self, seed_for_seeds=1):
        super(SynchronizeController, self).__init__()
        self.main_worker = None
        self.seed_generator = numpy.random.RandomState(seed_for_seeds)

    def handle_control(self, req, worker_id):
        if req == 'is_main_worker?':
            print 'is_main_worker?', worker_id
            if not self.main_worker:
                self.main_worker = worker_id
            result = self.main_worker == worker_id
            print result
            return result
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
