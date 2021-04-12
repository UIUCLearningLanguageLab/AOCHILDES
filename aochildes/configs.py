from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    transcripts = root / 'original_transcripts'
