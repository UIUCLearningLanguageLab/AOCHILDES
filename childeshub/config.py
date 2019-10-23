from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    data = src.parent / 'original_transcripts'
    items = src / 'corpora'


class Symbols:
    OOV = 'OOV'
    TITLED = 'TITLED'
    all = [OOV, TITLED, 'xxx']