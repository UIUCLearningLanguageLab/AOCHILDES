import random
from typing import Set, List, Optional, Tuple
from functools import reduce
from operator import iconcat

from aochildes.params import ChildesParams
from aochildes.transcripts import Transcripts
from aochildes.processor import PostProcessor


def split_into_sentences(tokens: List[str],
                         punctuation: Set[str],
                         ) -> List[List[str]]:
    assert isinstance(punctuation, set)

    res = [[]]
    for n, w in enumerate(tokens):
        res[-1].append(w)
        if w in punctuation and n < len(tokens) - 1:  # prevent appending empty list at end
            res.append([])
    return res


def chunk_sentences(sentences: List[List[str]],
                    split_size: int,
                    ):
    for i in range(0, len(sentences), split_size):
        yield sentences[i:i + split_size]


def tokens_from_docs(docs: List[str],
                     ) -> List[str]:
    tokenized_docs = [d.split() for d in docs]
    tokens = reduce(iconcat, tokenized_docs, [])  # flatten list of lists
    return tokens


class ChildesDataSet:
    def __init__(self,
                 params: Optional[ChildesParams] = None,
                 ):

        if params is None:
            params = ChildesParams()

        self.transcripts = Transcripts(params)
        self.processor = PostProcessor(params)

    def load_tokens(self):
        raise NotImplementedError  # TODO use this as input to preppy

    def load_docs(self,
                  shuffle_docs: Optional[bool] = False,
                  shuffle_sentences: Optional[bool] = False,
                  num_test_docs: Optional[int] = 0,
                  shuffle_seed: Optional[int] = 20,
                  split_seed: Optional[int] = 3,
                  remove_symbols: Optional[List[str]] = None,
                  ) -> Tuple[List[str], List[str]]:

        docs = self.processor.process(self.transcripts.age_ordered)
        num_docs = len(docs)
        print(f'Loaded {num_docs} transcripts')

        # shuffle at sentence-level (as opposed to document-level)
        # this remove clustering of same-age utterances within documents
        if shuffle_sentences:

            # TODO test

            random.seed(shuffle_seed)
            print('WARNING: Shuffling sentences')
            tokens = tokens_from_docs(docs)
            sentences = split_into_sentences(tokens, punctuation={'.', '!', '?'})
            random.shuffle(sentences)
            # assign sentences to documents + convert back to strings
            num_s_in_doc = len(sentences) // num_docs
            print(f'Found {len(sentences)} sentences')
            print(f'Assigning {num_s_in_doc} sentences to each new document')
            docs = [' '.join([w for s in s_chunk for w in s]) for s_chunk in chunk_sentences(sentences, num_s_in_doc)]
            print(f'After shuffling, number of docs={len(docs)}')

        if remove_symbols is not None:
            for n in range(num_docs):
                for symbol in remove_symbols:
                    docs[n] = docs[n].replace(f'{symbol} ', '')

        # split train/test
        print('Splitting docs into train and test...')
        num_test_doc_ids = num_docs - num_test_docs
        random.seed(split_seed)
        test_doc_ids = random.sample(range(num_test_doc_ids), num_test_docs)
        test_docs = []
        for test_line_id in test_doc_ids:
            test_doc = docs.pop(test_line_id)  # removes line and returns removed line
            test_docs.append(test_doc)

        # shuffle after train/test split
        if shuffle_docs:
            print('Shuffling documents')
            random.seed(shuffle_seed)
            random.shuffle(docs)

        print(f'Collected {len(docs):,} train docs')
        print(f'Collected {len(test_docs):,} test docs')

        return docs, test_docs
