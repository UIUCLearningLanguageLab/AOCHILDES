import attr


@attr.s
class Params:
    max_days = attr.ib(default=365 * 6)
    collection_names = attr.ib(default=['Eng-NA'])
    bad_speaker_roles = attr.ib(default=['Target_Child', 'Child'])
    min_utterance_length = attr.ib(default=1)
    normalize_names = attr.ib(default=True)
    normalize_titles = attr.ib(default=False)
    punctuation = attr.ib(default=True)

    merge_collocations = attr.ib(default=True)
    replace_archaic_words = attr.ib(default=True)
    replace_slang = attr.ib(default=True)
    handle_spacy_contractions = attr.ib(default=True)
    distinguish_possessive = attr.ib(default=False)