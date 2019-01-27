
class Params:
    def __init__(self):
        self.params = {'mb_size': 64,
                       'num_iterations': 20,
                       'num_parts': 2,
                       'bptt_steps': 7,
                       'num_saves': 10,
                       'part_order': 'inc_age',
                       'num_types': 4096,
                       'corpus_name': 'childes-20180319',
                       'sem_probes_name': 'childes-20180319_4096',
                       'syn_probes_name': 'childes-20180319_4096'}  # TODO use JW list - use date naming convention

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            raise AttributeError("No such attribute: " + name)

