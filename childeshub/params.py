import attr


@attr.s
class HubParams:
    mb_size = attr.ib(default= 64)
    num_iterations = attr.ib(default= [20, 20])
    num_parts = attr.ib(default= 2)
    bptt_steps = attr.ib(default= 7)
    num_saves = attr.ib(default= 10)
    part_order = attr.ib(default= 'inc_age')
    shuffle_docs = attr.ib(default= False)
    num_types = attr.ib(default= 4096)
    corpus_name = attr.ib(default= 'childes-20180319')
    probes_name = attr.ib(default= 'childes-20180319_4096')


@attr.s
class ItemParams:
    max_days = attr.ib(default=365 * 6)
    collection_names = attr.ib(default=['Eng-NA'])
    bad_speaker_roles = attr.ib(default=['Target_Child', 'Child'])
    min_utterance_length = attr.ib(default=1)
    lowercase = attr.ib(default=True)
    normalize_titles = attr.ib(default=False)
    punctuation = attr.ib(default=True)