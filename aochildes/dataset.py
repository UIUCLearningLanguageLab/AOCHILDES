
from typing import Set, List, Optional, Tuple
from functools import reduce
from operator import iconcat

from aochildes.params import ChildesParams
from aochildes.transcripts import Transcripts
from aochildes.processor import PostProcessor


def tokens_from_transcripts(transcripts: List[str],
                            ) -> List[str]:
    tokenized_transcripts = [d.split() for d in transcripts]
    tokens = reduce(iconcat, tokenized_transcripts, [])  # flatten list of lists
    return tokens


class ChildesDataSet:
    def __init__(self,
                 params: Optional[ChildesParams] = None,
                 ):

        if params is None:
            params = ChildesParams()

        self.transcripts = Transcripts(params)
        self.processor = PostProcessor(params)

    def load_processed_transcripts(self,
                                   ) -> List[str]:

        res = self.processor.process(self.transcripts.age_ordered)
        print(f'Loaded {len(res)} processed transcripts')

        return res

    def load_tokens(self) -> List[str]:
        ts = self.load_processed_transcripts()
        res = tokens_from_transcripts(ts)
        return res

    def load_text(self) -> str:
        return ' '.join(self.load_tokens())
