
from typing import List
import pyprind


from aochildes.params import ChildesParams
from aochildes.spelling import w2w


class PostProcessor:
    def __init__(self, params=None, verbose=False):
        self.params = params or ChildesParams()
        self.verbose = verbose

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
                # split compounds
                if self.params.split_compounds:
                    w = w.replace('+', ' ').replace('-', ' ').replace('_', ' ')
                # normalize speaker codes
                if w == 'chi' or w == 'Chi':
                    w = 'child'
                elif w == 'mot' or w == 'Mot':
                    w = 'mother'
                elif w == 'fat' or w == 'Fat':
                    w = 'father'

                words.append(w)

            line = ' '.join(words)
            lines.append(line)
            progress_bar.update()

        return lines
