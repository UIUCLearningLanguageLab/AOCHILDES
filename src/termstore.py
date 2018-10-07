from collections import Counter, OrderedDict
from cached_property import cached_property
from sortedcontainers import SortedSet
import numpy as np
from itertools import islice
import random
from itertools import chain

from configs import GlobalConfigs


def make_terms(configs_dict):
    def load_lines(item_name):
        f = GlobalConfigs.SRC_DIR / 'rnnlab' / 'items' / '{}_{}.txt'.format(configs_dict['corpus_name'], item_name)
        res = f.open('r').readlines()
        return res

    def split_lines(l, ids):
        test_lines = []
        for test_line_id in ids:
            test_line = l.pop(test_line_id)  # removes line and returns removed line
            test_lines.append(test_line)
        res = (list(chain(*[line.split() for line in l])),
               list(chain(*[line.split() for line in test_lines])))
        return res

    # split train test
    train_raws = {}
    test_raws = {}
    for name in ['terms', 'tags']:
        lines = load_lines(name)
        num_test_line_ids = len(lines) - GlobalConfigs.NUM_TEST_LINES  # adjust because of popping
        random.seed(3)
        test_line_ids = random.sample(range(num_test_line_ids), GlobalConfigs.NUM_TEST_LINES)
        train_raw, test_raw = split_lines(lines, test_line_ids)
        # oov
        train_set = set(train_raw)
        for n, item in enumerate(test_raw):
            if item not in train_set:
                test_raw[n] = GlobalConfigs.OOV_SYMBOL
        train_raws[name] = train_raw
        test_raws[name] = test_raw
    # terms
    train_terms = TermStore(train_raws['terms'], train_raws['tags'], configs_dict)
    test_terms = TermStore(test_raws['terms'], test_raws['tags'], configs_dict, types=train_terms.types)
    result = train_terms, test_terms
    return result


class TermStore(object):
    """
    Stores terms.
    """

    def __init__(self, terms, tags, configs_dict, types=None):
        self.num_types = configs_dict['num_types']
        self.mb_size = configs_dict['mb_size']
        self.bptt_steps = configs_dict['bptt_steps']
        self.num_y = configs_dict['num_y']
        self.p_noise = configs_dict['p_noise']
        self.f_noise = configs_dict['f_noise']
        self.block_order = configs_dict['block_order']
        self.num_parts = configs_dict['num_parts']
        self._types = types  # test items must have train types otherwise ids don't match
        self.tokens_no_oov, self.tags_no_oov = self.preprocess(terms, tags)

    # /////////////////////////////////////////////////// items

    def preprocess(self, terms, tags):
        raw_items = list(zip(terms, tags))
        if self._types is None:
            items = self.add_f_noise(self.prune(self.add_p_noise(raw_items)))  # train
        else:
            items = self.prune(raw_items)  # test
        result = list(zip(*items))
        return result

    def make_item_length(self, num_raw, max_num_docs=GlobalConfigs.MAX_NUM_DOCS):
        """
        Find length by which to prune items such that result is divisible by num_docs and
        such that the result of this division must be divisible by mb_size
        after first subtracting num_items_in_window.
        One cannot simply add num_items_in_window*num_docs because this might give result that is
        larger than number of available items.
        One can't use num_items_in_window to calculate the factor, because num_items_in_window
        should only be added once only to each document
        """
        # factor
        num_items_in_window = self.bptt_steps + self.num_y
        factor = self.mb_size * (max_num_docs if self._types is None else 1) + num_items_in_window
        # make divisible
        num_factors = num_raw // factor
        result = num_factors * factor - ((num_factors - (self.num_parts if self._types is None else 1))
                                         * num_items_in_window)
        return result

    def prune(self, p_noised):
        num_p_noised = len(p_noised)
        item_length = self.make_item_length(num_p_noised)
        pruned = p_noised[:item_length]
        print('Pruned {:,} total items to {:,}'.format(num_p_noised, item_length))
        return pruned

    def add_p_noise(self, raw):
        """
        Inserts P_NOISE at periodic intervals. Interval can be set to change gradually over corpus.
        IMPORTANT: Gradual change is relative to corpus order not block_order.
        """
        if 'no' in self.p_noise:
            return raw
        #
        interval = int(self.p_noise.split('_')[-1])
        num_pruned = len(raw)
        probs = np.zeros(num_pruned)
        if 'late' in self.p_noise:
            probs[::interval] = np.linspace(0.0, 1.0, 1 + (num_pruned - 1) // interval)
        elif 'early' in self.p_noise:
            probs[::interval] = np.linspace(1.0, 0.0, 1 + (num_pruned - 1) // interval)
        elif 'all' in self.p_noise:
            probs[::interval] = np.linspace(0.5, 0.5, 1 + (num_pruned - 1) // interval)
        else:
            raise AttributeError('rnnlab: Did not recognize arg to "p_noise".')
        # insert p_noise
        result = []
        for item, prob in zip(raw, probs):
            if random.random() < prob:
                result.append((GlobalConfigs.P_NOISE_SYMBOL, GlobalConfigs.P_NOISE_SYMBOL))
            result.append(item)
        return result

    def add_f_noise(self, p_noised):
        if self.f_noise == 0:
            return p_noised
        else:
            freq_d = {item: 0 for item in set(p_noised)}
            result = []
            for item in p_noised:
                if freq_d[item] >= self.f_noise:
                    result.append(item)
                else:
                    result.append((GlobalConfigs.F_NOISE_SYMBOL, GlobalConfigs.F_NOISE_SYMBOL))
                    freq_d[item] += 1
            return result

    # /////////////////////////////////////////////////// terms

    @cached_property
    def type_freq_dict_no_oov(self):
        c = Counter(self.tokens_no_oov)
        result = OrderedDict(
            sorted(c.items(), key=lambda item: (item[1], item[0]), reverse=True))  # order matters
        return result

    @cached_property
    def types(self):
        if self._types is None:
            most_freq_items = list(islice(self.type_freq_dict_no_oov.keys(), 0, self.num_types))
            sorted_items = sorted(most_freq_items[:-1] + [GlobalConfigs.OOV_SYMBOL])
            result = SortedSet(sorted_items)
        else:
            result = self._types
        return result

    @cached_property
    def term_id_dict(self):
        result = {item: n for n, item in enumerate(self.types)}
        return result

    @cached_property
    def tokens(self):
        result = []
        for token in self.tokens_no_oov:
            if token in self.term_id_dict:
                result.append(token)
            else:
                result.append(GlobalConfigs.OOV_SYMBOL)
        return result

    @cached_property
    def token_ids(self):
        result = [self.term_id_dict[token] for token in self.tokens]
        return result

    @cached_property
    def oov_id(self):
        result = self.term_id_dict[GlobalConfigs.OOV_SYMBOL]
        return result

    @cached_property
    def num_tokens(self):
        result = len(self.tokens)
        return result

    @cached_property
    def term_freq_dict(self):
        result = Counter(self.tokens)
        return result

    @cached_property
    def term_tags_dict(self):
        assert len(self.tokens_no_oov) == len(self.tags_no_oov)
        tag_set = set(self.tags_no_oov)
        result = {term: {tag: 0 for tag in tag_set}
                  for term in self.types}
        for term, tag in zip(self.tokens, self.tags_no_oov):
            result[term][tag] += 1
        return result
