from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    original = src / 'original_transcripts'
    corpora = root / 'corpora'


class Symbols:
    NAME = '[NAME]'
    PLACE = '[PLACE]'
    MISC = '[MISC]'

    child_name = '[CHILD_NAME]'
    mother_name = '[MOTHER_NAME]'
    father_name = '[FATHER_NAME]'


