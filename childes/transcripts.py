import pandas as pd
import numpy as np
from cached_property import cached_property
import datetime
import spacy
import pyprind
import attr
import yaml
import re
from pathlib import Path
from typing import List, Optional

from childes import config
from childes.normalize import w2w
from childes.params import Params
from childes.mergers import PersonMerger, PlacesMerger


col2dtype = {'id': np.int,
             'speaker_role': str,
             'gloss': str,
             'type': str,
             'num_tokens': np.int,
             'transcript_id': np.int,
             'target_child_age': np.float,
             'target_child_sex': str,
             'collection_name': str}

# make sure to add spaces after each utterance boundary marker
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


class Transcripts:

    def __init__(self, params=None, sex=None):
        self.params = params or Params()

        # load each utterance as a row in original_transcripts frame
        dfs = [pd.read_csv(csv_path,
                           index_col='id',
                           usecols=col2dtype.keys(),
                           dtype=col2dtype)
               for csv_path in sorted(config.Dirs.original.glob('*.csv'))]
        self.df = pd.DataFrame(pd.concat(dfs))

        # drop rows
        print('Transcripts: Utterances before dropping rows: {:>8,}'.format(len(self.df)))
        self.df.drop(self.df[self.df['target_child_age'] > self.params.max_days].index, inplace=True)
        self.df.drop(self.df[self.df['num_tokens'] < self.params.min_utterance_length].index, inplace=True)
        self.df.drop(self.df[self.df['speaker_role'].isin(self.params.bad_speaker_roles)].index, inplace=True)
        self.df.drop(self.df[~self.df['collection_name'].isin(self.params.collection_names)].index, inplace=True)
        print('Transcripts: Utterances after  dropping rows: {:>8,}'.format(len(self.df)))

        if sex:
            self.df.drop(self.df[self.df['target_child_sex'] != sex].index, inplace=True)
            print('Transcripts: Utterances after  filter by sex: {:>8,}'.format(len(self.df)))

        self._ages = []

    @cached_property
    def age_ordered(self):

        ignore_regex = re.compile(r'(�|www|xxx|yyy)')

        res = []
        for age, rows in self.df.groupby('target_child_age'):
            for transcript_id, rows2 in rows.groupby('transcript_id'):

                transcript = ''
                for gloss, utterance_type in zip(rows2['gloss'], rows['type']):
                    if ignore_regex.findall(gloss):
                        continue

                    transcript += gloss
                    if self.params.punctuation:
                        transcript += punctuation_dict[utterance_type]

                res.append(transcript)
                self._ages.append(age)

        return res

    @cached_property
    def ages(self):
        _ = self.age_ordered
        return self._ages

    @property
    def num_transcripts(self):
        return len(self.age_ordered)


class PostProcessor:
    def __init__(self, params=None, verbose=False):
        self.params = params or Params()
        self.verbose = verbose

    def normalize(self, w):
        # spelling
        if self.params.normalize_spelling and w.lower_ in w2w:
            return w2w[w.lower_]

        if w.is_title:
            # names
            if w._.is_person and self.params.normalize_persons:
                res = config.Symbols.NAME
            # places
            elif w._.is_place and self.params.normalize_places:
                res = config.Symbols.PLACE
            else:
                res = w.text
        else:
            res = w.lower_

        return res  # don't lowercase here otherwise symbols are affected

    @staticmethod
    def fix_childes_coding(line):
        line = re.sub(r' Chi ', f' {config.Symbols.NAME} ', line)
        line = re.sub(r' Mot ', f' {config.Symbols.NAME} ', line)
        line = re.sub(r' Fat ', f' {config.Symbols.NAME} ', line)
        return line

    @staticmethod
    def fix_spacy_tokenization(line):
        line = re.sub(r'valentine \'s day', 'valentines_day', line)
        line = re.sub(r'valentine \'s', 'valentines', line)
        line = re.sub(r'guy \'s', 'guys', line)
        line = re.sub(r'mommy\'ll', 'mommy will', line)
        line = re.sub(r'daddy\'ll', 'mommy will', line)
        line = re.sub(r'this\'ll', 'this will', line)
        line = re.sub(r'cann\'t', 'can not', line)
        line = re.sub(r' let \'s building', r' let us build', line)
        line = re.sub(r' let \'s looking', r' let us look', line)
        return line

    @staticmethod
    def replace_archaic_words(line):
        line = re.sub(r' oatios', ' oats', line)
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
        nlp.add_pipe(person_merger, last=True)
        nlp.add_pipe(places_merger, last=True)

        lines = []
        for doc in nlp.pipe(transcripts, batch_size=batch_size, disable=['tagger', 'parser', 'ner']):
            line = ' '.join([self.normalize(word) for word in doc])

            # some small fixes
            line = self.fix_childes_coding(line)
            line = self.fix_spacy_tokenization(line)
            line = self.replace_archaic_words(line) if self.params.replace_archaic_words else line

            print(line)
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
            output_path = config.Dirs.corpora
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
