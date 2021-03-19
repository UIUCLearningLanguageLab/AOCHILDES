
from typing import List, Optional

import pyprind


from childes import configs
from childes.params import ChildesParams
from childes.spelling import w2w
from childes.tokenization import special_cases


class PostProcessor:
    def __init__(self, params=None, verbose=False):
        self.params = params or ChildesParams()
        self.verbose = verbose

    def normalize(self, w: str):
        # spelling
        if self.params.normalize_spelling and w.lower() in w2w:
            return w2w[w.lower()]

        # case
        else:
            if self.params.lowercase:
                res = w.lower()
            else:
                res = w

        return res  # don't lowercase here otherwise symbols are affected

    @staticmethod
    def fix_childes_coding(line: str) -> str:
        new_words = []
        for w in line.split():
            if w == 'chi' or w == 'Chi':
                w = configs.Symbols.child_name
            elif w == 'mot' or w == 'Mot':
                w = configs.Symbols.mother_name
            elif w == 'fat' or w == 'Fat':
                w = configs.Symbols.father_name
            new_words.append(w)
        return ' '.join(new_words)

    def process(self,
                transcripts: List[str],
                ):
        """
        input is a list of unprocessed transcripts (each transcript is a string).
        output is a list of processed transcripts
        """

        num_transcripts = len(transcripts)
        print('Processor: Processing {} transcripts...'.format(num_transcripts))
        progress_bar = pyprind.ProgBar(num_transcripts)

        lines = []
        for doc in transcripts:
            line = ' '.join([self.normalize(word) for word in doc])

            # TODO special_cases tokenization rules

            # some small fixes
            line = self.fix_childes_coding(line)

            lines.append(line)
            progress_bar.update()

        return lines
