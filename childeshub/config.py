from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    analysis = src.parent / 'analysis'
    data = src.parent / 'data'
    items = src / 'items'
    probes = src / 'probes'


class Probes:
    verbose = True  # print probes not in vocab


class Terms:
    NUM_TEST_LINES = 100
    MAX_NUM_DOCS = 2048
    OOV_SYMBOL = 'OOV'
    SPECIAL_SYMBOLS = [OOV_SYMBOL] + ['xxx', 'TITLED', 'NAME_B', 'NAME_I']
    pos2tags = {'verb': ['BES', 'HVS', 'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                'noun': ['NN', 'NNS', 'WP'],
                'adverb': ['EX', 'RB', 'RBR', 'RBS', 'WRB'],
                'pronoun': ['PRP'],
                'preposition': ['IN'],
                'conjunction': ['CC'],
                'interjection': ['UH'],
                'determiner': ['DT'],
                'particle': ['POS', 'RP', 'TO'],
                'punctuation': [',', ':', '.', "''", 'HYPH', 'LS', 'NFP'],
                'adjective': ['AFX', 'JJ', 'JJR', 'JJS', 'PDT', 'PRP$', 'WDT', 'WP$'],
                'special': []}


class Hub:
    random_seed = 0