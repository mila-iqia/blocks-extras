import logging

from blocks.extensions import SimpleExtension
from platoon.channel import Soldier, Lieutenant

logger = logging.getLogger(__name__)

class Synchronize(SimpleExtension):
    def __init__(self, job_name, control_port, sync_rule, **kwargs):
        kwargs.setdefault("before_training", True)
        super(Synchronize, self).__init__(**kwargs)

        self.job_name = job_name
        self.sync_rule = sync_rule
        self.soldier = Soldier(cport=control_port, socket_timeout=2000)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['soldier']
        return state

    def do(self, which_callback, *args):
        if which_callback == 'before_training':
            should_initialize = self.soldier.send_req('init?')
            self.soldier.init_shared_params(
                self.job_name, self.main_loop.model.parameters,
                self.sync_rule, cleanup=should_initialize)
            if not should_initialize:
                for param, shared_param in zip(self.main_loop.model.parameters,
                                               self.soldier.shared_params):
                    param.set_value(shared_param)
                logger.debug("Copied parameters from shared")
            else:
                logger.debug("Initialized shared parameters")
        elif (which_callback == 'after_batch' or
              which_callback == 'after_epoch'):
            self.soldier.sync_params(synchronous=True)


class SynchronizeLieutenant(Lieutenant):

    def __init__(self):
        super(SynchronizeLieutenant, self).__init__()
        self.parameters_initilized = False

    def handle_control(self, req):
        print req
        if req == 'init?':
            if self.parameters_initilized:
                return False
            else:
                self.parameters_initilized = True
                return True

if __name__ == '__main__':
    import sys
    lieutenant = SynchronizeLieutenant()
    lieutenant.init_control(int(sys.argv[1]))
    lieutenant.serve()
