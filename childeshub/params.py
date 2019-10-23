import attr


@attr.s
class ItemParams:
    max_days = attr.ib(default=365 * 6)
    collection_names = attr.ib(default=['Eng-NA'])
    bad_speaker_roles = attr.ib(default=['Target_Child', 'Child'])
    min_utterance_length = attr.ib(default=1)
    lowercase = attr.ib(default=True)
    normalize_titles = attr.ib(default=False)
    punctuation = attr.ib(default=True)
