
class Params:
    def __init__(self):
        self.params = {'mb_size': 64,
                       'num_iterations': [20, 20],
                       'num_parts': 2,
                       'bptt_steps': 7,
                       'num_saves': 10,
                       'part_order': 'inc_age',
                       'shuffle_docs': False,
                       'num_types': 4096,
                       'corpus_name': 'childes-20180319',
                       'probes_name': 'childes-20180319_4096'}

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            raise AttributeError("No such attribute: " + name)

