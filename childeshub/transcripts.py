import pandas as pd
import numpy as np
from cached_property import cached_property
import datetime
import spacy
import pyprind
import yaml

from childeshub import config
from childeshub.params import ItemParams

nlp = spacy.load('en_core_web_sm')

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

    def __init__(self, params=None):
        self.params = params or ItemParams()

        # load each utterance as a row in data frame
        dfs = [pd.read_csv(csv_path,
                           index_col='id',
                           usecols=col2dtype.keys(),
                           dtype=col2dtype)
               for csv_path in sorted(config.Dirs.data.glob('*.csv'))]
        self.df = pd.concat(dfs)

        # drop rows
        print('Transcripts: Utterances before dropping rows: {}'.format(len(self.df)))
        self.df.drop(self.df[self.df['target_child_age'] > self.params.max_days].index, inplace=True)
        self.df.drop(self.df[self.df['num_tokens'] < self.params.min_utterance_length].index, inplace=True)
        self.df.drop(self.df[self.df['speaker_role'].isin(self.params.bad_speaker_roles)].index, inplace=True)
        self.df.drop(self.df[~self.df['collection_name'].isin(self.params.collection_names)].index, inplace=True)
        print('Transcripts: Utterances after  dropping rows: {}'.format(len(self.df)))

        self._sexes = []  # keep track of sex when making transcripts

    @cached_property
    def age_ordered(self):
        res = []
        for age, rows in self.df.groupby('target_child_age'):
            for transcript_id, rows2 in rows.groupby('transcript_id'):

                # keep track of sex
                sexes = rows['target_child_sex'].dropna().tolist()
                if not len(set(sexes)) == 1:
                    self._sexes.append('na')
                else:
                    self._sexes.append(sexes[0])  # there should only be 1 sex (1 child) per transcript

                transcript = ''
                for gloss, utterance_type in zip(rows2['gloss'], rows['type']):
                    transcript += gloss
                    if self.params.punctuation:
                        transcript += punctuation_dict[utterance_type]
                res.append(transcript)
        return res

    @property
    def sexes(self):
        _ = self.age_ordered
        return self._sexes

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
            normalized = 'TITLED'
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

    def to_file(self, terms_list, tags_list, path_to_folder=None, dry_run=False, first_only=False):
        print('Processor: Writing to disk...')
        date_str = datetime.datetime.now().strftime('%Y%m%d')
        corpus_name = 'childes-{}'.format(date_str)

        if path_to_folder is None:
            path_to_folder = config.Dirs.items
        if dry_run:
            path_to_folder = config.Dirs.items / 'dry_runs'

        params_path = path_to_folder / '{}_{}.yaml'.format(corpus_name, 'params')
        terms_path = path_to_folder / '{}_{}.txt'.format(corpus_name, 'terms')
        tags_path = path_to_folder / '{}_{}.txt'.format(corpus_name, 'tags')

        f1 = terms_path.open('w', encoding='utf-8')
        f2 = tags_path.open('w', encoding='utf-8')

        for terms, tags in zip(terms_list, tags_list):
            f1.write(' '.join(terms) + '\n')
            f2.write(' '.join(tags) + '\n')
            if first_only:
                break

        f1.close()
        f2.close()

        # save params
        with params_path.open('w', encoding='utf8') as f:
            yaml.dump(self.params.__dict__, f, default_flow_style=False, allow_unicode=True)
