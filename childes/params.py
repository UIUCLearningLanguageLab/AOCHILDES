import attr


@attr.s
class Params:
    max_days = attr.ib(default=365 * 6)
    collection_names = attr.ib(default=['Eng-NA'])
    bad_speaker_roles = attr.ib(default=['Target_Child', 'Child'])
    min_utterance_length = attr.ib(default=1)
    punctuation = attr.ib(default=True)
    exclude_unknown_utterances = attr.ib(default=True)
    lowercase = attr.ib(default=True)

    normalize_persons = attr.ib(default=False)
    normalize_places = attr.ib(default=False)
    normalize_misc = attr.ib(default=False)
    normalize_spelling = attr.ib(default=False)