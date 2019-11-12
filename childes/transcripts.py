import pandas as pd
import numpy as np
from cached_property import cached_property
import datetime
import spacy
import pyprind
import attr
import yaml
from pathlib import Path
from typing import List, Optional

from childes import config
from childes.params import ItemParams


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
        self.params = params or ItemParams()

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
        res = []
        for age, rows in self.df.groupby('target_child_age'):
            for transcript_id, rows2 in rows.groupby('transcript_id'):

                transcript = ''
                for gloss, utterance_type in zip(rows2['gloss'], rows['type']):
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
        self.params = params or ItemParams()
        self.verbose = verbose

    def normalize(self, word, words_excluded_from_ner):
        if len(word.text) <= 2:
            normalized = word.text if not self.params.lowercase else word.text.lower()
        elif word.text.lower() in words_excluded_from_ner:  # prevent probes from potentially being treated as NEs
            normalized = word.text if not self.params.lowercase else word.text.lower()
        elif word.text.istitle() and self.params.normalize_titles:
            normalized = config.Symbols.TITLED
        else:
            normalized = word.text if not self.params.lowercase else word.text.lower()

        if word.text != normalized and self.verbose:
            print('{:<12} -> {:<12}'.format(word.text, normalized))
        return normalized

    def process(self, transcripts, excluded_words=None, tagging=True, batch_size=100):
        """
        input is a list of unprocessed transcripts (each transcript is a string).
        output is a list of tokens and part-of-speech tags (each transcript is a list)
        """

        if excluded_words is None:
            excluded_words = []

        num_transcripts = len(transcripts)
        print('Processor: Processing {} transcripts...'.format(num_transcripts))
        progress_bar = pyprind.ProgBar(num_transcripts)

        disabled = ['tagger', 'parser', 'ner']
        if self.params.normalize_titles:
            disabled.remove('ner')
        if tagging:
            disabled.remove('tagger')

        nlp = spacy.load('en_core_web_sm')  # do not put in outer scope; might raise un-necessary OSError instead

        terms_list = []
        tags_list = []
        for doc in nlp.pipe(transcripts, batch_size=batch_size, disable=disabled):
            terms = [self.normalize(word, excluded_words) for word in doc]
            tags = [word.tag_ for word in doc]
            assert len(terms) == len(tags)
            terms_list.append(terms)
            tags_list.append(tags)
            progress_bar.update()

        if not tagging:
            return terms_list
        else:
            return terms_list, tags_list

    def to_file(self,
                terms_list: List[str],
                tags_list: List[str],
                ages: List[float],
                output_dir: Optional[str] = None,
                suffix: str = ''):
        print('Processor: Writing to disk...')
        date_str = datetime.datetime.now().strftime('%Y%m%d')
        corpus_name = 'childes-{}'.format(date_str)

        if output_dir is None:
            output_path = Path('output')
        else:
            output_path = Path(output_dir)

        if not output_path.exists():
            output_path.mkdir()

        params_path = output_path / '{}_{}{}.yaml'.format(corpus_name, 'params', suffix)
        terms_path = output_path / '{}_{}{}.txt'.format(corpus_name, 'terms', suffix)
        tags_path = output_path / '{}_{}{}.txt'.format(corpus_name, 'tags', suffix)
        ages_path = output_path / '{}_{}{}.txt'.format(corpus_name, 'ages', suffix)

        f1 = terms_path.open('w', encoding='utf-8')
        f2 = tags_path.open('w', encoding='utf-8')
        f3 = ages_path.open('w', encoding='utf-8')

        for terms, tags, age in zip(terms_list, tags_list, ages):
            f1.write(' '.join(terms) + '\n')
            f2.write(' '.join(tags) + '\n')
            f3.write(str(age) + '\n')

        f1.close()
        f2.close()

        # save params
        with params_path.open('w', encoding='utf8') as f:
            yaml.dump(attr.asdict(self.params), f, default_flow_style=False, allow_unicode=True)
