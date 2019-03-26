import numpy as np
import pyprind
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
import matplotlib.pyplot as plt
from scipy import stats

from childeshub.hub import Hub

HUB_MODE = 'sem'
NUM_PCS = 30


NGRAM_RANGE = (2, 2)  # TODO
CORRELATE = False
COMPARE_PROBES_ONLY = False

LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (20, 6)
DPI = None
YLIM = 20

EXCLUDED = []  # ['may', 'june', 'sweet', 'back', 'behind', 'fly']  # TODO

hub = Hub(mode=HUB_MODE)

START1, END1 = 0, hub.midpoint_loc // 8
START2, END2 = hub.train_terms.num_tokens - END1, hub.train_terms.num_tokens

CATS = [] or hub.probe_store.cats  #  ['bug', 'dessert', 'body', 'bug']


def make_in_out_corr_mat(start, end):
    print('Making in_out_corr_mat with start={} and end={}'.format(start, end))
    tokens = hub.train_terms.tokens[start:end]
    # types
    types = sorted(set(tokens))
    num_types = len(types)
    type2id = {t.lower(): n for n, t in enumerate(types)}  # OOV needs to be lower-case because of ngram pipeline
    # ngrams
    ngrams = hub.get_ngrams(NGRAM_RANGE, tokens)
    ngram_types = sorted(set(ngrams))
    num_ngram_types = len(ngram_types)
    ngram2id = {t: n for n, t in enumerate(ngram_types)}
    # cooc_mat
    cooc_mat = np.zeros((num_ngram_types, num_types))
    print('Co-occurrence matrix shape={}'.format(cooc_mat.shape))
    pbar = pyprind.ProgBar(num_types)
    for n, ngram in enumerate(ngrams[:-NGRAM_RANGE[-1]]):  # TODO test
        row_id = ngram2id[ngram]
        next_ngram = ngrams[n + 1]
        next_type = next_ngram.split()[-1]
        try:
            col_id = type2id[next_type]
        except KeyError:  # TODO not sure

            # print(next_ngram)
            # print(next_type)
            # print()

            continue
        cooc_mat[row_id, col_id] += 1
        pbar.update()
    # corr_mat

    # TODO
    # res = cosine_similarity(cooc_mat) if CORRELATE else cooc_mat
    res = manhattan_distances(cooc_mat) if CORRELATE else cooc_mat
    return res, ngram_types


def any_word_in_ngram(ngram, words):
    for word in words:
        if word in ngram:
            return True
    else:
        return False


def calc_t(id2ngram, words, u_col):  # TODO speed this up by computing ids which to compare once and use for all PCs
    ref = []
    exp = []
    for ngram_id, val in enumerate(u_col):
        ngram = id2ngram[ngram_id]
        if any_word_in_ngram(ngram, words):
            exp.append(val)
        elif any_word_in_ngram(ngram, hub.probe_store.types):
            ref.append(val)
        else:
            pass  # skip non-probes
    # t-statistic

    print(len(ref))  # TODO test that ngrams work
    print(len(exp))

    t, pval = stats.ttest_ind(ref, exp, equal_var=False)
    return abs(t)


def calc_t_at_each_pc(pc_mat, ngramtypes, words):
    id2ngram = {n: t for n, t in enumerate(ngramtypes)}
    res = []
    for pc_id, u_col in enumerate(pc_mat.T):
        metric = calc_t(id2ngram, words, u_col)
        res.append(metric)
    return res


# make in_out_corr_mats
label1 = 'start={:,}&end={:,}'.format(START1, END1)
label2 = 'start={:,}&end={:,}'.format(START2, END2)
in_out_corr_mat1, ngram_types1 = make_in_out_corr_mat(START1, END1)
in_out_corr_mat2, ngram_types2 = make_in_out_corr_mat(START2, END2)

# fig for each category
for cat in CATS:
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('Do any of the first {} PCs of cooc mat (dist={})\ncode for the category {}?'.format(
        NUM_PCS, NGRAM_RANGE[-1], cat.upper()), fontsize=AX_FONTSIZE)
    ax.set_ylabel('Abs(t)', fontsize=AX_FONTSIZE)
    ax.set_xlabel('Principal Component', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.set_ylim([0, YLIM])
    # plot critical t
    df = hub.probe_store.num_probes - 2  # df when equal var is assumed
    p = 1 - 0.05 / NUM_PCS
    crit_t = stats.t.ppf(p, df)
    ax.axhline(y=crit_t, color='grey')
    # plot
    for label, corr_mat, ngram_types in [(label1, in_out_corr_mat1, ngram_types1),
                                         (label2, in_out_corr_mat2, ngram_types2)]:
        # pca
        pca = PCA(n_components=NUM_PCS)  # = u, s, v
        u = pca.fit_transform(corr_mat)  # (num_types, NUM_PCS)
        cat_probes = [w for w in hub.probe_store.cat_probe_list_dict[cat] if w not in EXCLUDED]
        y = calc_t_at_each_pc(u, ngram_types, cat_probes)
        #
        ax.plot(y, label=label, linewidth=2)
    ax.legend(loc='best', frameon=False, fontsize=LEG_FONTSIZE,
              bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()