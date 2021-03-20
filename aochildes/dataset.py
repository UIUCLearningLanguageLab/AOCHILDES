
from typing import Set, List, Optional, Tuple
from functools import reduce
from operator import iconcat

from aochildes.params import ChildesParams
from aochildes.pipeline import Pipeline
from aochildes.helpers import Transcript


def tokens_from_transcripts(transcripts: List[str],
                            ) -> List[str]:
    tokenized_transcripts = [d.split() for d in transcripts]
    tokens = reduce(iconcat, tokenized_transcripts, [])  # flatten list of lists
    return tokens


def split_into_sentences(tokens: List[str],
                         ) -> List[List[str]]:

    res = [[]]
    for n, w in enumerate(tokens):
        res[-1].append(w)
        if w.endswith('.') or w.endswith('?') or w.endswith('!') and n < len(tokens) - 1:  # prevent  empty list at end
            res.append([])
    return res


class ChildesDataSet:
    def __init__(self,
                 params: Optional[ChildesParams] = None,
                 ):

        if params is None:
            params = ChildesParams()

        self.pipeline = Pipeline(params)
        self.transcripts: List[Transcript] = self.pipeline.load_age_ordered_transcripts()

    def load_tokens(self) -> List[str]:
        res = tokens_from_transcripts([t.text for t in self.transcripts])
        return res

    def load_sentences(self) -> List[str]:
        res = []
        for t in self.transcripts:
            res.extend(t.sentences)
        return res

    def load_text(self) -> str:
        return ' '.join(self.load_tokens())
