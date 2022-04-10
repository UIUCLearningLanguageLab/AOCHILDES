from typing import List
from dataclasses import dataclass, field


@dataclass
class AOChildesParams:
    max_days: int = field(default=365 * 6)
    collection_names: List[str] = field(default_factory=lambda: ['Eng-NA'])
    bad_speaker_roles: List[str] = field(default_factory=lambda: ['Target_Child', 'Child'])
    min_utterance_length: int = field(default=1)
    punctuation: bool = field(default=True)
    exclude_unknown_utterances: bool = field(default=True)

    normalize_spelling: bool = field(default=True)
    split_compounds: bool = field(default=True)  # convert "+" and "-" to "_"
