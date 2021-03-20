import pandas as pd
import re
from typing import List
import pyprind

from aochildes import configs
from aochildes.helpers import Transcript, col2dtype, punctuation_dict
from aochildes.params import ChildesParams
from aochildes.spelling import w2w


class Pipeline:

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

    def process(self,
                sentence: str,
                ) -> str:

        words = []
        for w in sentence.split():
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

        return ' '.join(words)

    def load_age_ordered_transcripts(self) -> List[Transcript]:

        print('Preparing AOCHILDES transcripts...')
        pbar = pyprind.ProgBar(len(self.df.groupby('target_child_age')), stream=1)

        ignore_regex = re.compile(r'(�|www|xxx|yyy)')

        res = []
        for age, rows in self.df.groupby('target_child_age'):
            for transcript_id, rows2 in rows.groupby('transcript_id'):

                sentences = []
                for gloss, utterance_type in zip(rows2['gloss'], rows['type']):

                    # exclude bad sentence
                    if self.params.exclude_unknown_utterances:
                        if ignore_regex.findall(gloss):
                            continue

                    # add utterance boundary marker
                    if self.params.punctuation:
                        gloss += f' {punctuation_dict[utterance_type]}'

                    processed_sentence = self.process(gloss)
                    sentences.append(processed_sentence)

                res.append(Transcript(sentences, age))

            pbar.update()

        return res
