import pandas as pd
import numpy as np
from cached_property import cached_property
import re

from aochildes import configs
from aochildes.params import ChildesParams

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
        self.params = params or ChildesParams()

        # load each utterance as a row in original_transcripts frame
        dfs = [pd.read_csv(csv_path,
                           index_col='id',
                           usecols=col2dtype.keys(),
                           dtype=col2dtype)
               for csv_path in sorted(configs.Dirs.original.glob('*.csv'))]
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
                    if self.params.exclude_unknown_utterances:

                        if ignore_regex.findall(gloss):
                            continue

                    transcript += gloss
                    if self.params.punctuation:
                        transcript += f' {punctuation_dict[utterance_type]}'

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


