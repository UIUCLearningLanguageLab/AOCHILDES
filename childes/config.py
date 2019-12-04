from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    original = src / 'original_transcripts'
    corpora = root / 'corpora'


class Symbols:
    NAME = '[NAME]'
    PLACE = '[PLACE]'


