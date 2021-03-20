
from typing import List, Optional

import pyprind


from aochildes import configs
from aochildes.params import ChildesParams
from aochildes.spelling import w2w


class PostProcessor:
    def __init__(self, params=None, verbose=False):
        self.params = params or ChildesParams()
        self.verbose = verbose

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
            words = []
            for w in doc.split():
                # always lower-case
                w = w.lower()
                # fix spelling
                if self.params.normalize_spelling and w in w2w:
                    w = w2w[w.lower()]
                # normalize compounds
                if self.params.normalize_compounds:
                    w = w.replace('+', '_').replace('-', '_')

                words.append(w)

            line = ' '.join(words)

            # some small fixes
            line = self.fix_childes_coding(line)

            lines.append(line)
            progress_bar.update()

        return lines
