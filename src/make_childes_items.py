from probestore import ProbeStore
from options import default_configs_dict

import os
import csv
import itertools
import datetime
import spacy
import pyprind

DRY_RUN = True

CSV_DIR_NAME = 'childesr_csvs'
SORT_CRITERION = 'target_child_age'
BAD_SPEAKER_ROLES = ['Target_Child', 'Child']
COLLECTION_NAMES = ['Eng-NA']
MAX_DAYS = 365 * 6
MIN_NUM_TOKENS_IN_UTT = 1

ENT_TYPES = []  # ['ORG', 'PERSON']  # normalize members if not empty
BAD_ENT_TYPES = []  # ['DATE', 'TIME', 'LOC', 'PRODUCT']  # do not normalize members
LOWER_CASE = True
NORMALIZE_TITLES = False
PUNCTUATION = False

VERBOSE_SORT = False
VERBOSE_NORMALIZE = False
VERBOSE_VALIDATION = False


def is_valid(d):
    if not d['target_child_age'].split('.')[0].isnumeric():
        if VERBOSE_VALIDATION:
            print('Age not available.')
        return False
    if float(d['target_child_age']) > MAX_DAYS:
        if VERBOSE_VALIDATION:
            print('Age larger than {} days.'.format(MAX_DAYS))
        return False
    if int(d['num_tokens']) < MIN_NUM_TOKENS_IN_UTT:
        if VERBOSE_VALIDATION:
            print('No tokens in utterance.')
        return False
    if d['speaker_role'] in BAD_SPEAKER_ROLES:
        if VERBOSE_VALIDATION:
            print('Bad speaker_role.')
        return False
    if d['collection_name'] not in COLLECTION_NAMES:
        if VERBOSE_VALIDATION:
            print('Bad collection_name.')
        return False
    return True


def gen_transcripts(ds, criterion):
    transcript = ''
    for d in ds:
        if not is_valid(d):
            continue
        transcript_id = d['transcript_id']
        d_criterion = d[criterion]
        while d['transcript_id'] == transcript_id:
            if is_valid(d):
                transcript += to_utterance(d)
            try:
                d = next(ds)
            except StopIteration:
                break  # do not remove
        yield (d_criterion, transcript)
        transcript = to_utterance(d)


def to_utterance(d):
    if PUNCTUATION:
        utterance = '{}{} '
        punctation_dict = {'imperative': '!', 'question': '?'}
        try:
            punctuation = punctation_dict[d['type'].split('_')[0]]
        except KeyError:
            punctuation = '.'
        return utterance.format(d['gloss'], punctuation)
    else:
        utterance = '{} '
        return utterance.format(d['gloss'])


def normalize(word, probe_set):
    if len(word.text) <= 2:
        normalized = word.text if not LOWER_CASE else word.text.lower()
    elif word.text.lower() in probe_set:
        normalized = word.text if not LOWER_CASE else word.text.lower()
    elif word.ent_type_ in BAD_ENT_TYPES:
        normalized = word.text if not LOWER_CASE else word.text.lower()
    elif word.ent_type_ in ENT_TYPES:
        normalized = 'NAME_B' if word.ent_iob_ == 'B' else 'NAME_I'
    elif word.text.istitle() and NORMALIZE_TITLES:
        normalized = 'TITLED'
    else:
        normalized = word.text if not LOWER_CASE else word.text.lower()
    if word.text != normalized and VERBOSE_NORMALIZE:
        print('"{}" -> "{}"'.format(word.text, normalized))
    return normalized


def main():
    # get transcripts
    csvs_dir = os.path.join(os.path.expanduser('~'), 'Dropbox', CSV_DIR_NAME)
    csv_paths = sorted([os.path.join(csvs_dir, f_name)
                        for f_name in os.listdir(csvs_dir) if f_name.endswith('csv')])
    readers = [csv.DictReader(open(csv_path, 'r')) for csv_path in csv_paths]
    chained_readers = itertools.chain(*readers)
    d_cs, ts = zip(*sorted(
        gen_transcripts(chained_readers, criterion=SORT_CRITERION), key=lambda tup: float(tup[0])))
    if VERBOSE_SORT:
        print(d_cs)

    # files
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    corpus_name = 'childes-{}'.format(date_str) if not DRY_RUN else 'dry_run'
    items_dir = os.path.join('rnnlab', 'items')
    terms_f, tags_f = [open(os.path.join(items_dir, '{}_{}.txt'.format(corpus_name, item_name)),
                            'w', encoding='utf-8') for item_name in ['terms', 'tags']]

    # process + export transcripts
    nlp = spacy.load('en_core_web_sm', disable=['parser'])
    probe_store = ProbeStore(default_configs_dict, hub_mode='sem')
    num_ts = len(ts)
    print('Processing {} transcripts...'.format(num_ts))
    pbar = pyprind.ProgBar(num_ts)
    for doc in nlp.pipe(ts):
        pbar.update()
        terms = [normalize(word, probe_store.types) for word in doc]
        tags = [word.tag_ for word in doc]
        for item_f, item_name, items in [(terms_f, 'terms', terms),
                                         (tags_f, 'tags', tags)]:
            item_f.write(' '.join(items) + '\n')


if __name__ == '__main__':
    main()