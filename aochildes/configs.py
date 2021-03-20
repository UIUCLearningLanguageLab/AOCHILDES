from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    original = root / 'original_transcripts'
    corpora = root / 'corpora'


class Symbols:

    child_name = '<child_name>'
    mother_name = '<mother_name>'
    father_name = '<father_name>'


