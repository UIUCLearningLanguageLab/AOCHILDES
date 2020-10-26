import datetime
import re
from pathlib import Path
from typing import List, Optional

import attr
import pyprind
import spacy
import yaml

from childes import configs
from childes.mergers import PersonMerger, PlacesMerger, MiscMerger
from childes.params import Params
from childes.spelling import w2w
from childes.tokenization import special_cases


class PostProcessor:
    def __init__(self, params=None, verbose=False):
        self.params = params or Params()
        self.verbose = verbose

    def normalize(self, w):
        # spelling
        if self.params.normalize_spelling and w.lower_ in w2w:
            return w2w[w.lower_]

        # persons
        if w._.is_person and self.params.normalize_persons:
            res = configs.Symbols.NAME
        # places
        elif w._.is_place and self.params.normalize_places:
            res = configs.Symbols.PLACE
        # miscellaneous
        elif w._.is_misc and self.params.normalize_misc:
            res = configs.Symbols.MISC
        else:
            if self.params.lowercase:
                res = w.lower_
            else:
                res = w.text

        return res  # don't lowercase here otherwise symbols are affected

    @staticmethod
    def fix_childes_coding(line):
        line = re.sub(r' Chi ', f' child ', line)
        line = re.sub(r' Mot ', f' mother ', line)
        line = re.sub(r' Fat ', f' father ', line)
        return line

    def process(self, transcripts, batch_size=100):
        """
        input is a list of unprocessed transcripts (each transcript is a string).
        output is a list of processed transcripts
        """

        num_transcripts = len(transcripts)
        print('Processor: Processing {} transcripts...'.format(num_transcripts))
        progress_bar = pyprind.ProgBar(num_transcripts)

        nlp = spacy.load('en_core_web_sm')  # do not put in outer scope; might raise un-necessary OSError instead
        person_merger = PersonMerger(nlp)
        places_merger = PlacesMerger(nlp)
        misc_merger = MiscMerger(nlp)
        nlp.add_pipe(person_merger, last=True)
        nlp.add_pipe(places_merger, last=True)
        nlp.add_pipe(misc_merger, last=True)
        for s, rule in special_cases:
            nlp.tokenizer.add_special_case(s, rule)

        lines = []
        for doc in nlp.pipe(transcripts, batch_size=batch_size, disable=['tagger', 'parser', 'ner']):
            line = ' '.join([self.normalize(word) for word in doc])

            # some small fixes
            line = self.fix_childes_coding(line)

            lines.append(line)
            progress_bar.update()

        return lines

    def to_file(self,
                lines: List[str],
                ages: List[float],
                output_dir: Optional[str] = None,
                suffix: str = ''):
        print('Processor: Writing to disk...')
        date_str = datetime.datetime.now().strftime('%Y%m%d')
        corpus_name = 'childes-{}'.format(date_str)

        if output_dir is None:
            output_path = configs.Dirs.corpora
        else:
            output_path = Path(output_dir)

        if not output_path.exists():
            output_path.mkdir()

        params_path = output_path / '{}_{}{}.yaml'.format(corpus_name, 'params', suffix)
        terms_path = output_path / '{}_{}{}.txt'.format(corpus_name, 'terms', suffix)
        ages_path = output_path / '{}_{}{}.txt'.format(corpus_name, 'ages', suffix)

        f1 = terms_path.open('w', encoding='utf-8')
        f2 = ages_path.open('w', encoding='utf-8')

        for line, age in zip(lines, ages):
            f1.write(line + '\n')
            f2.write(str(age) + '\n')

        f1.close()
        f2.close()

        # save params
        with params_path.open('w', encoding='utf8') as f:
            yaml.dump(attr.asdict(self.params), f, default_flow_style=False, allow_unicode=True)