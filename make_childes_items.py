import csv
import itertools
import datetime
import spacy
import pyprind

from childeshub.probestore import ProbeStore
from childeshub.params import Params
from childeshub import config

DRY_RUN = True

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

VERBOSE_SORT = True
VERBOSE_NORMALIZE = False
VERBOSE_VALIDATION = False

HUB_MODE = 'sem'  # prevent semantic probes from being normalized


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


def gen_transcripts(csv_readers, criterion):
    transcript = ''
    for reader in csv_readers:
        if not is_valid(reader):
            continue
        transcript_id = reader['transcript_id']
        value = reader[criterion]
        while reader['transcript_id'] == transcript_id:
            if is_valid(reader):
                transcript += to_utterance(reader)
            try:
                reader = next(csv_readers)
            except StopIteration:
                break  # do not remove
        yield (value, transcript)
        transcript = to_utterance(reader)


def to_utterance(d):
    if PUNCTUATION:
        utterance = '{}{} '
        punctuation_dict = {'imperative': '!', 'question': '?'}
        try:
            punctuation = punctuation_dict[d['type'].split('_')[0]]
        except KeyError:
            punctuation = '.'
        return utterance.format(d['gloss'], punctuation)
    else:
        utterance = '{} '
        return utterance.format(d['gloss'])


def normalize(word, probe_set):
    if len(word.text) <= 2:
        normalized = word.text if not LOWER_CASE else word.text.lower()
    elif word.text.lower() in probe_set:  # prevent probes from potentially being treated as NEs
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
    csv_paths = sorted(config.Dirs.data.glob('*.csv'))
    readers = [csv.DictReader(csv_path.open('r')) for csv_path in csv_paths]
    chained_readers = itertools.chain(*readers)
    criterion_values, transcripts = zip(*sorted(
        gen_transcripts(chained_readers, criterion=SORT_CRITERION), key=lambda tup: float(tup[0])))
    if VERBOSE_SORT:
        print(criterion_values)

    # files
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    corpus_name = 'childes-{}'.format(date_str) if not DRY_RUN else 'dry_run'
    terms_file_name = config.Dirs.items / '{}_{}.txt'.format(corpus_name, 'terms')
    tags_file_name = config.Dirs.items / '{}_{}.txt'.format(corpus_name, 'tags')
    f1 = terms_file_name.open('w', encoding='utf-8')
    f2 = tags_file_name.open('w', encoding='utf-8')

    nlp = spacy.load('en_core_web_sm', disable=['parser'])
    probe_store = ProbeStore(HUB_MODE, Params().probes_name)

    # process + export transcripts
    num_transcripts = len(transcripts)
    print('Processing {} transcripts...'.format(num_transcripts))
    pbar = pyprind.ProgBar(num_transcripts)
    for doc in nlp.pipe(transcripts):
        pbar.update()
        terms = [normalize(word, probe_store.types) for word in doc]
        tags = [word.tag_ for word in doc]
        f1.write(' '.join(terms) + '\n')
        f2.write(' '.join(tags) + '\n')


if __name__ == '__main__':
    main()