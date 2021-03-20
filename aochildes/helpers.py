from dataclasses import dataclass, field
from typing import List


@dataclass
class Transcript:
    sentences: List[str]
    age: float
    text: str = field(init=False)

    def __post_init__(self):
        self.text = ' '.join(self.sentences)


col2dtype = {'id': int,
             'speaker_role': str,
             'gloss': str,
             'type': str,
             'num_tokens': int,
             'transcript_id': int,
             'target_child_age': float,
             'target_child_sex': str,
             'collection_name': str}
punctuation_dict = {'imperative': '! ',
                    'imperative_emphatic': '! ',
                    'question exclamation': '! ',
                    'declarative': '. ',
                    'interruption': '. ',
                    'self interruption': '. ',
                    'quotation next line': '. ',
                    'quotation precedes': '. ',
                    'broken for coding': '. ',
                    'question': '? ',
                    'self interruption question': '? ',
                    'interruption question': '? ',
                    'trail off question': '? ',
                    'trail off': '. '}