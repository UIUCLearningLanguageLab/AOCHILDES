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

    replace_archaic_words = attr.ib(default=True)
    normalize_spelling = attr.ib(default=True)
    prettify_spacy_contractions = attr.ib(default=False)
    distinguish_possessive = attr.ib(default=False)