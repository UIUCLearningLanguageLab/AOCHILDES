
class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class HubParams:
    mb_size = 64  # TODO make public interface to edit this
    num_iterations = 20
    num_parts = 2
    bptt_steps = 7
    num_y = 1
    num_saves = 10
    part_order = 'inc_age'
    sem_probes_name = 'semantic-raw'
    syn_probes_name = 'syntactic-mcdi'


default_hub_params = ObjectView({k: v for k, v in HubParams.__dict__.items() if not k.startswith('_')})