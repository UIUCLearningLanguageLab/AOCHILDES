
class Params(dict):
    def __init__(self):
        super().__init__()
        self.params = {'mb_size': 64,
                       'num_iterations': 20,
                       'num_parts': 2,
                       'bptt_steps': 7,
                       'num_y': 1,
                       'num_saves': 10,
                       'part_order': 'inc_age',
                       'p_noise': 'no_0',
                       'f_noise': 0,
                       'num_types': 4096,
                       'corpus_name': 'childes-20180319',
                       'sem_probes_name': 'semantic-raw',
                       'syn_probes_name': 'syntactic-mcdi'}

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            raise AttributeError("No such attribute: " + name)

