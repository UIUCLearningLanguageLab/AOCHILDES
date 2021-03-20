from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    original = root / 'original_transcripts'
    corpora = root / 'corpora'
