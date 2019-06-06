from cached_property import cached_property
import numpy as np
from itertools import chain
from scipy.signal import lfilter
import pyprind
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from scipy.stats import linregress
import pandas as pd
import random
import scipy.stats
import string
from cytoolz import itertoolz

from childeshub.probestore import ProbeStore
from childeshub.termstore import make_terms
from childeshub.params import Params
from childeshub import config


class CachedAndModeSwitchable(object):
    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, hub, cls):
        hub.cached_property_names.append(self.func.__name__)
        value = hub.__dict__[self.func.__name__] = self.func(hub)
        return value


class Hub(object):
    def __init__(self, mode='sem', terms=(None, None), params=None, **kwargs):
        self.mode = mode
        self.params = params or self.make_params(kwargs)
        self._terms = terms
        self.cached_property_names = []  # names of cached properties that need to be invalidated after mode switching
        self.mode2probestore = {}

    # ////////////////////////////////////////////// init

    @cached_property
    def train_terms(self):
        result = self._terms[0] or make_terms(self.params)[0]
        return result

    @cached_property
    def test_terms(self):
        result = self._terms[1] or make_terms(self.params)[1]
        return result

    @CachedAndModeSwitchable
    def probe_store(self):
        try:
            return self.mode2probestore[self.mode]
        except KeyError:
            res = ProbeStore(self.mode, self.params.probes_name, self.train_terms.term_id_dict)
            self.mode2probestore[self.mode] = res
            return res

    @staticmethod
    def make_params(kwargs):
        if kwargs is None:
            raise RuntimeError('No params passed to hub.')
        res = Params()
        for k, v in kwargs.items():
            if k not in res.params:
                raise KeyError('"{}" not in Hub Params.'.format(k))
            else:
                res.params[k] = v
        return res

    def switch_mode(self, mode):
        # invalidate cached properties
        for property_name in sorted(self.__dict__.keys()):
            if property_name in self.cached_property_names:
                del self.__dict__[property_name]
                print('Deleted cached "{}"'.format(property_name))
        # switch mode
        self.mode = mode
        print('Switched hub mode to "{}"'.format(mode))

    # ////////////////////////////////////////////// info

    @cached_property
    def num_mbs_in_part(self):
        result = self.num_windows_in_part / self.params.mb_size
        assert result.is_integer()
        return int(result)

    @cached_property
    def num_mbs_in_test(self):
        result = self.num_windows_in_test / self.params.mb_size
        assert result.is_integer()
        return int(result)

    @cached_property
    def num_iterations_list(self):
        result = np.linspace(self.params.num_iterations[0], self.params.num_iterations[1],
                             num=self.params.num_parts, dtype=np.int)
        return result

    @cached_property
    def mean_num_iterations(self):
        result = np.mean(self.num_iterations_list)
        return result

    @cached_property
    def num_mbs_in_block(self):
        result = self.num_mbs_in_part * self.mean_num_iterations
        return result

    @cached_property
    def num_mbs_in_token_ids(self):
        result = self.num_mbs_in_part * self.params.num_parts
        return int(result)

    @cached_property
    def num_items_in_window(self):
        num_items_in_window = self.params.bptt_steps + 1
        return num_items_in_window

    @cached_property
    def num_windows_in_part(self):
        result = self.num_items_in_part - self.num_items_in_window
        return result

    @cached_property
    def num_windows_in_test(self):
        result = self.test_terms.num_tokens - self.num_items_in_window  # TODO test
        return result

    @cached_property
    def num_items_in_part(self):
        result = self.train_terms.num_tokens / self.params.num_parts
        assert float(result).is_integer()
        return int(result)

    @cached_property
    def stop_mb(self):
        stop_mb = self.params.num_parts * self.num_mbs_in_block
        return stop_mb

    @cached_property
    def data_mbs(self):
        mbs_in_timepoint = int(self.stop_mb / self.params.num_saves)
        end = mbs_in_timepoint * self.params.num_saves + mbs_in_timepoint
        data_mbs = list(range(0, end, mbs_in_timepoint))
        return data_mbs

        # ////////////////////////////////////////////////////////// partitions

    def partition(self, token_ids):  # partitions should always contain token_ids rather than tokens
        result = []
        for i in range(0, len(token_ids), self.num_items_in_part):
            result.append(token_ids[i:i + self.num_items_in_part])
        return result

    def order_pseudo_randomly(self, parts):
        idx = []
        for i, j in zip(np.roll(np.arange(self.params.num_parts), self.params.num_parts // 2)[::2],
                        np.roll(np.arange(self.params.num_parts), self.params.num_parts // 2)[::-2]):
            idx += [i, j]
        assert len(set(idx)) == len(parts)
        res = [parts[i] for i in idx]
        assert len(res) == len(parts)
        return res

    def calc_num_pos_in_part(self, pos, part, num_pos=1024):
        assert isinstance(part[0], int)
        # make pos_term_ids
        if pos in config.Terms.pos2tags.keys():
            pos_term_ids = [self.train_terms.term_id_dict[term] for term in getattr(self, pos + 's')][:num_pos]
        elif '+' in pos:
            pos_term_ids = []
            for pos in pos.split('+'):
                pos_term_ids += [self.train_terms.term_id_dict[term] for term in getattr(self, pos + 's')][:num_pos]
        else:
            pos_term_ids = [self.train_terms.term_id_dict[probe] for probe in self.probe_store.types]
        # result
        result = len([1 for term_id in part if term_id in pos_term_ids])
        return result

    def calc_num_unique_ngrams_in_part(self, ngram_range, part):
        v = CountVectorizer(ngram_range=ngram_range)
        a = v.build_analyzer()
        tokens = [self.train_terms.types[term_id] for term_id in part]
        ngrams = a(' '.join(tokens))  # requires string
        result = len(set(ngrams))
        return result

    def calc_part_probes_context_stat(self, sort_by, part, max_context_term_ids=1000):
        # check_term_ids
        if 'probes' in sort_by:
            check_term_ids = [self.train_terms.term_id_dict[probe] for probe in self.probe_store.types]
        elif 'nouns' in sort_by:
            check_term_ids = [self.train_terms.term_id_dict[noun] for noun in self.nouns]
        elif 'punctuations' in sort_by:
            check_term_ids = [self.train_terms.term_id_dict[punct] for punct in self.punctuations]
        elif 'prepositions' in sort_by:
            check_term_ids = [self.train_terms.term_id_dict[prep] for prep in self.prepositions]
        elif 'conjunctions' in sort_by:
            check_term_ids = [self.train_terms.term_id_dict[conj] for conj in self.conjunctions]
        elif 'pronouns' in sort_by:
            check_term_ids = [self.train_terms.term_id_dict[pron] for pron in self.pronouns]
        elif 'verbs' in sort_by:
            check_term_ids = [self.train_terms.term_id_dict[v] for v in self.verbs]
        else:
            raise AttributeError('rnnlab: Invalid arg to "sort_by".')
        # context_size
        if '1' in sort_by:
            context_size = 1
        elif '2' in sort_by:
            context_size = 2
        elif '3' in sort_by:
            context_size = 3
        elif '4' in sort_by:
            context_size = 4
        else:
            raise AttributeError('Invalid arg to "sort_by".')
        #
        is_right = True if 'right' in sort_by else False
        # context_term_ids
        context_term_ids = []
        for loc, term_id in enumerate(part):
            if term_id in check_term_ids:
                if not is_right:
                    start = max(0, loc - context_size)
                    for t_id in part[start: loc]:
                        context_term_ids.append(t_id)
                else:
                    end = max(0, loc + context_size)
                    for t_id in part[loc: end]:
                        context_term_ids.append(t_id)
                if len(context_term_ids) > max_context_term_ids:  # this eliminates confound of context frequency
                    break
        else:
            print('WARNING: Collected  {}/{} context_term_ids.'.format(
                len(context_term_ids), max_context_term_ids))
        # stats
        if 'context-set-size' in sort_by:
            result = len(set(context_term_ids))
        elif 'context-entropy' in sort_by:
            result = self.calc_entropy(context_term_ids)
        elif 'context-frequency' in sort_by:
            result = np.sum([self.train_terms.term_freq_dict[self.train_terms.types[term_id]]
                             for term_id in context_term_ids])
        else:
            raise AttributeError('Invalid arg to "sort_by".')
        return result

    def calc_part_id_sort_stat_dict(self, parts, sort_by):
        result = {}
        for part_id, part in enumerate(parts):
            if sort_by == 'age':
                sort_stat = part_id  # TODO test
            elif '+' in sort_by or sort_by in config.Terms.pos2tags.keys():
                sort_stat = self.calc_num_pos_in_part(sort_by, part)
            elif sort_by == 'entropy':
                sort_stat = self.calc_entropy(part)
            elif 'gram' in sort_by:
                ngram_size = int(sort_by[0])
                ngram_range = (ngram_size, ngram_size)
                sort_stat = self.calc_num_unique_ngrams_in_part(ngram_range, part)
            elif 'context' in sort_by:
                sort_stat = self.calc_part_probes_context_stat(sort_by, part)
            else:
                raise AttributeError('rnnlab: Invalid arg to "sort_by".')
            if np.isnan(sort_stat):
                print('rnnlab WARNING: Could not calculate {} statistic for part {}'.format(sort_by, part_id))
            result[part_id] = sort_stat
        return result

    @cached_property
    def reordered_partitions(self):
        result = self.reorder_parts()
        return result

    def reorder_parts(self, part_order=None):
        partitions = self.partition(self.train_terms.token_ids)
        if len(partitions) == 1:  # test set
            return partitions
        if part_order is None:
            part_order = self.params.part_order
        # sort
        order_by, sort_by = part_order.split('_')
        sort_d = self.calc_part_id_sort_stat_dict(partitions, sort_by)
        sorted_partitions = list(list(zip(*sorted(enumerate(partitions),
                                                  key=lambda i: sort_d[i[0]])))[1])  # need list not tuple
        # order by increasing, decreasing, etc
        if order_by == 'unordered':
            result = self.order_pseudo_randomly(sorted_partitions)
        elif order_by == 'shuffled':
            np.random.seed(3)  # determined seed based on equal probe density at start of training
            np.random.shuffle(sorted_partitions)  # do not make this into list
            result = sorted_partitions
        elif order_by == 'inc':
            result = sorted_partitions
        elif order_by == 'dec':
            result = sorted_partitions[::-1]
        elif order_by == 'middec':
            num_mid_parts = self.params.num_parts // 3
            start = (self.params.num_parts // 2) - (num_mid_parts // 2)
            end = start + num_mid_parts
            mid_parts = sorted_partitions[start: end]
            remaining_parts = sorted_partitions[:start] + sorted_partitions[end:]
            result = mid_parts + remaining_parts[::-1]
        elif order_by == 'midinc':
            num_mid_parts = self.params.num_parts // 3
            start = (self.params.num_parts // 2) - (num_mid_parts // 2)
            end = start + num_mid_parts
            mid_parts = sorted_partitions[start: end]
            remaining_parts = sorted_partitions[:start] + sorted_partitions[end:]
            result = mid_parts + remaining_parts
        else:
            raise AttributeError('rnnlab: Invalid arg to "order".')
        assert len(result) == len(partitions)
        return result

    @cached_property
    def reordered_token_ids(self):
        result = [token_id for partition in self.reordered_partitions for token_id in partition]
        return result

    @cached_property
    def reordered_tokens(self):
        result = [self.train_terms.types[token_id] for token_id in self.reordered_token_ids]
        return result

    @cached_property
    def first_half_tokens(self):
        midpoint = len(self.train_terms.tokens) // 2
        result = self.train_terms.tokens[:midpoint]
        return result

    @cached_property
    def second_half_tokens(self):
        midpoint = len(self.train_terms.tokens) // 2
        result = self.train_terms.tokens[-midpoint:]
        return result

    # ////////////////////////////////////////////// batching

    def make_windows_mat(self, part, num_windows):
        result = np.zeros((num_windows, self.num_items_in_window), dtype=np.int)
        for window_id in range(num_windows):
            window = part[window_id:window_id + self.num_items_in_window]
            result[window_id, :] = window
        return result

    def gen_ids(self, num_iterations_list=None, is_test=False):
        if not is_test:
            parts = self.reordered_partitions
            num_mbs_in_part = self.num_mbs_in_part
            num_windows = self.num_windows_in_part
        else:
            parts = [self.test_terms.token_ids]
            num_mbs_in_part = self.num_mbs_in_test
            num_windows = self.num_windows_in_test
        if not num_iterations_list:
            num_iterations_list = self.num_iterations_list
        # generate
        for part_id, part in enumerate(parts):
            windows_mat = self.make_windows_mat(part, num_windows)
            windows_mat_x, windows_mat_y = np.split(windows_mat, [self.params.bptt_steps], axis=1)
            num_iterations = num_iterations_list[part_id]
            print('Iterating {} times over part {}'.format(num_iterations, part_id))
            for _ in range(num_iterations):
                for x, y in zip(np.vsplit(windows_mat_x, num_mbs_in_part),
                                np.vsplit(windows_mat_y, num_mbs_in_part)):
                    yield x, y

    # ////////////////////////////////////////////// static

    @staticmethod
    def calc_entropy(labels):
        num_labels = len(labels)
        probs = np.asarray([count / num_labels for count in Counter(labels).values()])
        result = - probs.dot(np.log2(probs))
        return result

    @staticmethod
    def smooth(l, strength):
        b = [1.0 / strength] * strength
        a = 1
        result = lfilter(b, a, l)
        return result

    @staticmethod
    def roll_mean(l, size):
        result = pd.DataFrame(l).rolling(size).mean().values.flatten()
        return result

    def get_term_id_windows(self, term, roll_left=False, num_samples=64):
        locs = random.sample(self.term_unordered_locs_dict[term], num_samples)
        if not roll_left:  # includes term in window
            result = [self.train_terms.token_ids[loc - self.params.bptt_steps + 1: loc + 1]
                      for loc in locs if loc > self.params.bptt_steps]
        else:
            result = [self.train_terms.token_ids[loc - self.params.bptt_steps + 0: loc + 0]
                      for loc in locs if loc > self.params.bptt_steps]
        return result

    @staticmethod
    def split(l, split_size):
        for i in range(0, len(l), split_size):
            yield l[i:i + split_size]

    def make_locs_xy(self, terms, num_bins=20):
        item_locs_l = [self.term_unordered_locs_dict[i] for i in terms]  # TODO use reordered locs?
        locs_l = list(chain(*item_locs_l))
        hist, b = np.histogram(locs_l, bins=num_bins)
        result = (b[:-1], np.squeeze(hist))
        return result

    @staticmethod
    def fit_line(x, y):
        poly = np.polyfit(x, y, 1)
        result = np.poly1d(poly)(x)
        return result

    def calc_loc_asymmetry(self, term, num_bins=200):  # num_bins=200 for terms
        (x, y) = self.make_locs_xy([term], num_bins=num_bins)
        y_fitted = self.fit_line(x, y)
        result = linregress(x / self.train_terms.num_tokens, y_fitted)[0]  # divide x to increase slope
        return result

    @staticmethod
    def make_sentence_length_stat(items, is_avg, w_size=10000):
        # make sent_lengths
        last_period = 0
        sent_lengths = []
        for n, item in enumerate(items):
            if item in ['.', '!', '?']:
                sent_length = n - last_period - 1
                sent_lengths.append(sent_length)
                last_period = n
        # rolling window
        df = pd.Series(sent_lengths)
        if is_avg:
            print('Making sentence length rolling average...')
            result = df.rolling(w_size).std().values
        else:
            print('Making sentence length rolling std...')
            result = df.rolling(w_size).mean().values
        return result

    # ////////////////////////////////////////////// cached

    @cached_property
    def term_part_freq_dict(self):
        print('Making term_part_freq_dict...')
        result = {term: [0] * self.params.num_parts for term in self.train_terms.types}
        for part_id, part in enumerate(self.reordered_partitions):
            count_dict = Counter(part)
            for term_id, freq in count_dict.items():
                term = self.train_terms.types[term_id]
                result[term][part_id] = freq
        return result

    @cached_property
    def part_entropies(self):
        result = []
        for part in self.reordered_partitions:
            part_entropy = self.calc_entropy(part)
            result.append(part_entropy)
        return result

    @staticmethod
    def get_ngrams(n_gram_size, tokens):

        if not isinstance(n_gram_size, int):
            raise TypeError('This function was changed by PH in May 2019 because'
                            'previously used sklearn Countvectorizer uses stopwords'
                            ' and removes punctuation')
        ngrams = list(itertoolz.sliding_window(n_gram_size, tokens))
        return ngrams

    @CachedAndModeSwitchable
    def probe_x_mats(self, max_locs=2048, seed=config.Hub.random_seed):
        np.random.seed(seed=seed)
        result = []
        for probe in self.probe_store.types:
            try:
                locs = np.random.choice(self.term_unordered_locs_dict[probe], max_locs, replace=False)
            except ValueError:  # if replace=False and the sample size is greater than the population size
                locs = self.term_unordered_locs_dict[probe]
            probe_x_mat = np.asarray([self.train_terms.token_ids[loc + 1 - self.params.bptt_steps: loc + 1]
                                      for loc in locs if self.train_terms.num_tokens - 1 > loc >= self.params.bptt_steps])
            result.append(probe_x_mat)
        return result

    @CachedAndModeSwitchable
    def probe_y_mats(self, max_locs=2048, seed=config.Hub.random_seed):
        np.random.seed(seed=seed)
        result = []
        for probe in self.probe_store.types:
            try:
                locs = np.random.choice(self.term_unordered_locs_dict[probe], max_locs, replace=False)
            except ValueError:  # if replace=False and the sample size is greater than the population size
                locs = self.term_unordered_locs_dict[probe]
            probe_y_mat = np.asarray([[self.train_terms.token_ids[loc + 1]]
                                      for loc in locs if self.train_terms.num_tokens - 1 > loc >= self.params.bptt_steps])
            result.append(probe_y_mat)
        return result

    @CachedAndModeSwitchable
    def probe_num_periods_in_context_list(self):
        result = []
        for probe in self.probe_store.types:
            num_periods = self.probe_context_terms_dict[probe].count('.')
            result.append(num_periods / len(self.probe_context_terms_dict[probe]))
        return result

    @CachedAndModeSwitchable
    def probe_tag_entropy_list(self):
        result = []
        for probe in self.probe_store.types:
            tag_entropy = scipy.stats.entropy(list(self.train_terms.term_tags_dict[probe].values()))
            result.append(tag_entropy)
        return result

    @CachedAndModeSwitchable
    def probe_context_overlap_list(self):  # TODO use
        # probe_context_set_d
        probe_context_set_d = {probe: set() for probe in self.probe_store.types}
        for probe in self.probe_store.types:
            context_set = set(self.probe_context_terms_dict[probe])
            probe_context_set_d[probe] = context_set
        # probe_overlap_list
        result = []
        for probe in self.probe_store.types:
            set_a = probe_context_set_d[probe]
            num_overlap = 0
            for set_b in probe_context_set_d.values():
                num_overlap += len(set_a & set_b)
            probe_overlap = num_overlap / len(set_a)
            result.append(probe_overlap)
        return result

    @CachedAndModeSwitchable
    def probe_context_terms_dict(self):
        print('Making probe_context_terms_dict...')
        result = {probe: [] for probe in self.probe_store.types}
        for n, t in enumerate(self.train_terms.tokens):
            if t in self.probe_store.types:
                context = [term for term in self.train_terms.tokens[n - self.params.bptt_steps:n]]
                result[t] += context
        return result

    # ////////////////////////////////////////////// terms

    def get_most_frequent_terms(self, num_most_common):
        result = list(zip(*self.train_terms.term_freq_dict.most_common(num_most_common)))[0]
        return result

    def get_terms_related_to_cat(self, cat):
        cat_probes = self.probe_store.cat_probe_list_dict[cat]
        term_hit_dict = {term: 0 for term in self.train_terms.types}
        for n, term in enumerate(self.train_terms.tokens):
            loc = max(0, n - self.params.bptt_steps)
            if any([term in cat_probes for term in self.train_terms.tokens[loc: n]]):
                term_hit_dict[term] += 1
        result = list(zip(*sorted(term_hit_dict.items(),
                                  key=lambda i: i[1] / self.train_terms.term_freq_dict[i[0]])))[0]
        return result

    def get_term_set_prop_near_terms(self, terms, dist=1):
        ts = []
        for loc, t in enumerate(self.train_terms.tokens):
            if t in terms:
                try:
                    ts.append(self.train_terms.tokens[loc + dist])
                except IndexError:  # hit end or start of tokens
                    print('rnnlab: Invalid tokens location: {}'.format(loc + dist))
                    continue
        c = Counter(ts)
        result = sorted(set(ts), key=lambda t: c[t] / self.train_terms.term_freq_dict[t])
        return result

    def get_terms_near_term(self, term, dist=1):
        result = []
        for loc in self.term_reordered_locs_dict[term]:
            try:
                result.append(self.reordered_tokens[loc + dist])
            except IndexError:  # location too early or too late
                pass
        return result

    @CachedAndModeSwitchable
    def cat_common_successors_dict(self, num_most_common=60):  # TODO vary num_most_common
        cat_successors_dict = {cat: [] for cat in self.probe_store.cats}
        for cat, cat_probes in self.probe_store.cat_probe_list_dict.items():
            for cat_probe in cat_probes:
                terms_near_cat_probe = self.get_terms_near_term(cat_probe)
                cat_successors_dict[cat] += terms_near_cat_probe
        # filter
        result = {cat: list(zip(*Counter(cat_successors).most_common(num_most_common)))[0]
                  for cat, cat_successors in cat_successors_dict.items()}
        return result

    @CachedAndModeSwitchable
    def probes_common_successors_dict(self, num_most_common=60):  # TODO vary num_most_common
        probes_successors = []
        for probe in self.probe_store.types:
            terms_near_probe = self.get_terms_near_term(probe)
            probes_successors += terms_near_probe
        # filter
        result = list(zip(*Counter(probes_successors).most_common(num_most_common)))[0]
        return result

    @cached_property
    def nouns(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in config.Terms.pos2tags['noun']\
                    and term not in config.Terms.SPECIAL_SYMBOLS + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def adjectives(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in config.Terms.pos2tags['adjective']\
                    and term not in config.Terms.SPECIAL_SYMBOLS + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def verbs(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in config.Terms.pos2tags['verb']\
                    and term not in config.Terms.SPECIAL_SYMBOLS + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def adverbs(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in config.Terms.pos2tags['adverb']\
                    and term not in config.Terms.SPECIAL_SYMBOLS + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def pronouns(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in config.Terms.pos2tags['pronoun']\
                    and term not in config.Terms.SPECIAL_SYMBOLS + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def prepositions(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in config.Terms.pos2tags['preposition']\
                    and term not in config.Terms.SPECIAL_SYMBOLS + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def conjunctions(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in config.Terms.pos2tags['conjunction']\
                    and term not in config.Terms.SPECIAL_SYMBOLS + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def interjections(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in config.Terms.pos2tags['interjection']\
                    and term not in config.Terms.SPECIAL_SYMBOLS + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def determiners(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in config.Terms.pos2tags['determiner']\
                    and term not in config.Terms.SPECIAL_SYMBOLS + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def particles(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in config.Terms.pos2tags['particle']\
                    and term not in config.Terms.SPECIAL_SYMBOLS + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def punctuations(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in config.Terms.pos2tags['punctuation']\
                    and term not in config.Terms.SPECIAL_SYMBOLS + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def specials(self):
        result = [symbol for symbol in config.Terms.SPECIAL_SYMBOLS
                  if symbol in self.train_terms.types]
        return result

    # ////////////////////////////////////////////// lateness

    @cached_property
    def term_reordered_locs_dict(self):
        print('Making term_reordered_locs_dict...')
        result = {item: [] for item in self.train_terms.types}
        for loc, term in enumerate(self.reordered_tokens):
            result[term].append(loc)
        return result

    @cached_property
    def term_unordered_locs_dict(self):  # keep this for fast calculation where order doesn't matter
        print('Making term_unordered_locs_dict...')
        result = {item: [] for item in self.train_terms.types}
        for loc, term in enumerate(self.train_terms.tokens):
            result[term].append(loc)
        return result

    @cached_property
    def term_avg_reordered_loc_dict(self):
        result = {}
        for term, locs in self.term_reordered_locs_dict.items():
            result[term] = np.mean(locs)
        return result

    @cached_property
    def term_avg_unordered_loc_dict(self):
        result = {}
        for term, locs in self.term_unordered_locs_dict.items():
            result[term] = np.mean(locs)
        return result

    def calc_avg_reordered_loc(self, term):
        result = int(self.term_avg_reordered_loc_dict[term])
        return result

    def calc_avg_unordered_loc(self, term):
        result = int(self.term_avg_unordered_loc_dict[term])
        return result

    def calc_lateness(self, term, is_probe, reordered=True):
        fn = self.calc_avg_reordered_loc if reordered else self.calc_avg_unordered_loc
        if is_probe:
            ref_loc = self.probes_reordered_loc * 2 if reordered else self.probes_unordered_loc * 2
        else:
            ref_loc = self.train_terms.num_tokens
        result = round(fn(term) / ref_loc, 2)
        return result

    @CachedAndModeSwitchable
    def probe_lateness_dict(self):
        print('Making probe_lateness_dict...')
        probe_latenesses = []
        for probe in self.probe_store.types:
            probe_lateness = self.calc_lateness(probe, is_probe=True)
            probe_latenesses.append(probe_lateness)
        result = {probe: round(np.asscalar(np.mean(probe_lateness)), 2)
                  for probe_lateness, probe in zip(probe_latenesses, self.probe_store.types)}
        return result

    @CachedAndModeSwitchable
    def probes_reordered_loc(self):
        n_sum = 0
        num_ns = 0
        for n, term in enumerate(self.reordered_tokens):
            if term in self.probe_store.types:
                n_sum += n
                num_ns += 1
        result = n_sum / num_ns
        return result

    @CachedAndModeSwitchable
    def probes_unordered_loc(self):
        loc_sum = 0
        num_locs = 0
        for loc, term in enumerate(self.train_terms.tokens):
            if term in self.probe_store.types:
                loc_sum += loc
                num_locs += 1
        result = loc_sum / num_locs
        return result

    @property
    def midpoint_loc(self):
        result = self.train_terms.num_tokens // 2
        return result

    def split_probes_by_loc(self, num_splits, is_reordered=False):
        if is_reordered:
            d = self.term_avg_reordered_loc_dict
        else:
            d = self.term_avg_unordered_loc_dict
        probe_loc_pairs = [(probe, loc) for probe, loc in d.items()
                           if probe in self.probe_store.types]
        sorted_probe_loc_pairs = sorted(probe_loc_pairs, key=lambda i: i[1])
        num_in_split = self.probe_store.num_probes // num_splits
        for split_id in range(num_splits):
            start = num_in_split * split_id
            probes, _ = zip(*sorted_probe_loc_pairs[start: start + num_in_split])
            yield probes

    # ////////////////////////////////////////////// context goodness

    def calc_median_term_cg(self, term):
        result = np.median(self.make_term_cgs(term))
        return result

    @cached_property
    def loc_cf_dict(self):
        from bisect import bisect
        result = {loc: 0 for loc in range(self.train_terms.num_tokens)}
        pbar = pyprind.ProgBar(self.train_terms.num_tokens)
        for loc, term in enumerate(self.reordered_tokens):
            pbar.update()
            cf = bisect(self.term_reordered_locs_dict[term], loc)
            result[loc] += cf
        return result

    def make_term_cgs(self, term):  # this is fastest implementation
        result = []
        for term_loc in self.term_reordered_locs_dict[term]:
            term_loc = max(self.params.bptt_steps, term_loc)
            context_cf_sum = 0
            for loc in range(term_loc - self.params.bptt_steps, term_loc):
                context_cf_sum += self.loc_cf_dict[loc]
            cg = context_cf_sum / term_loc
            result.append(cg)
        return result

    # ////////////////////////////////////////////// context diversity

    @CachedAndModeSwitchable
    def probe_cds_dict(self):
        result = {}
        print('Making probe_cds_dict...')
        for probe in self.probe_store.types:
            result[probe] = self.make_term_cds(probe)
        return result

    def make_term_cds(self, term):
        result = []
        co_occurence_dict = {term: set()}
        for n, term_ in enumerate(self.reordered_tokens):
            if term_ == term:
                for context_term in self.reordered_tokens[n - self.params.bptt_steps:n]:
                    co_occurence_dict[term_].add(context_term)
                num_term_co_occurences = len(list(co_occurence_dict[term_]))
                context_diversity = num_term_co_occurences / self.train_terms.num_types
                result.append(context_diversity)
        return result
