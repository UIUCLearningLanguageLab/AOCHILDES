import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import linalg as slinalg
from scipy import sparse

from childeshub.hub import Hub

HUB_MODE = 'sem'
NUM_PCS = 100

COHEND = True

NOUNS = False  # TODO plot line for how diagnostic each pc is for nouns in general
NGRAM_SIZE = 2  # 3 will crash a 32GB RAM machine without sparse input matrix

LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (20, 6)
DPI = 200

EXCLUDED = []  # ['may', 'june', 'sweet', 'back', 'behind', 'fly']

hub = Hub(mode=HUB_MODE)

START1, END1 = 0, hub.midpoint_loc // 1
START2, END2 = hub.train_terms.num_tokens - END1, hub.train_terms.num_tokens

CATS = ['number', 'body', 'family'] or hub.probe_store.cats  # ['bug', 'dessert', 'body', 'bug']


def make_in_out_corr_mat(start, end):
    print('Making in_out_corr_mat with start={} and end={}'.format(start, end))
    tokens = hub.train_terms.tokens[start:end]
    # types
    types = sorted(set(tokens))
    num_types = len(types)
    type2id = {t.lower(): n for n, t in enumerate(types)}  # OOV needs to be lower-case because of ngram pipeline
    # ngrams
    ngrams = hub.get_ngrams((NGRAM_SIZE, NGRAM_SIZE), tokens)
    ngram_types = sorted(set(ngrams))
    num_ngram_types = len(ngram_types)
    ngram2id = {t: n for n, t in enumerate(ngram_types)}
    # make sparse matrix (types in rows, ngrams in cols)
    shape = (num_types, num_ngram_types)
    print('Making In-Out matrix with shape={}...'.format(shape))
    data = []
    rows = []
    cols = []
    for n, ngram in enumerate(ngrams[:-NGRAM_SIZE]):
        col_id = ngram2id[ngram]
        next_ngram = ngrams[n + 1]
        next_type = next_ngram.split()[-1]
        try:
            row_id = type2id[next_type]
        except KeyError:  # TODO contractions are removed from ngrams
            continue
        rows.append(row_id)
        cols.append(col_id)
        data.append(1)
    # make sparse matrix once (updating it is expensive)
    res = sparse.csr_matrix((data, (rows, cols)), shape=(num_types, num_ngram_types), dtype = np.int8)
    return res, types


def calc_cohend(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s


def calc_some_measure(exp_ids, ref_ids, u_col):
    ref_loadings = []
    exp_loadings = []
    # collect loadings
    for term_id, loading in enumerate(u_col):
        if term_id in exp_ids:
            exp_loadings.append(loading)
        elif term_id in ref_ids:
            ref_loadings.append(loading)
    # compare loadings of reference and experimental group
    assert exp_loadings
    assert ref_loadings
    if COHEND:
        return abs(calc_cohend(exp_loadings, ref_loadings))
    else:
        t, pval = stats.ttest_ind(ref_loadings, exp_loadings, equal_var=False)
        return abs(t)


def calc_measure_at_each_pc(pc_mat, types, words1, words2):  # TODO use words 2 to pit 2 cats against each other
    id2term = {n: t for n, t in enumerate(types)}
    # pre-compute ids at which either reference or experimental word is located
    ids_for_exp = set()
    ids_for_ref = set()
    print('Getting term_ids...')
    for term_id in range(pc_mat.shape[0]):
        term = id2term[term_id]
        if term in words1:
            ids_for_exp.add(term_id)
        else:
            ids_for_ref.add(term_id)
    #
    print('num ngrams with words1={}'.format(len(ids_for_exp)))
    print('num ngrams with words2={}'.format(len(ids_for_ref)))
    #
    res = []
    print('Calculating some measure for each PC...')
    for pc_id, u_col in enumerate(pc_mat.T):
        assert len(u_col) == pc_mat.shape[0]
        measure = calc_some_measure(ids_for_exp, ids_for_ref, u_col)
        res.append(measure)
    print(res)
    return res


def plot_comparison(y1, y2, cat, cum, color1='blue', color2='red'):
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('Do any of the first {} PCs of cooc mat (ngram size={})\ncode for the category {}?'.format(
        NUM_PCS, NGRAM_SIZE, cat.upper()), fontsize=AX_FONTSIZE)
    if COHEND:
        ax.set_ylabel('cumulative abs(Cohen d)' if cum else 'abs(Cohen d)', fontsize=AX_FONTSIZE)
    else:
        ax.set_ylabel('cumulative abs(t)' if cum else 'abs(t)', fontsize=AX_FONTSIZE)
    ax.set_xlabel('Principal Component', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    if not cum:
        ax.set_ylim([0, 2 if COHEND else 10])
    elif cum and COHEND:
        ax.set_ylim([0, 20])
    # plot critical t
    df = hub.probe_store.num_probes - 2  # df when equal var is assumed
    p = 1 - 0.05 / NUM_PCS
    crit_t = stats.t.ppf(p, df)
    if not cum and not COHEND:
        ax.axhline(y=crit_t, color='grey')
    # plot
    if cum:
        ax.plot(np.cumsum(y1), label='cum. ' + label1, linewidth=2, color=color1)
        ax.plot(np.cumsum(y2), label='cum. ' + label2, linewidth=2, color=color2)
    else:
        ax.plot(y1, label=label1, linewidth=2, color=color1)
        ax.plot(y2, label=label2, linewidth=2, color=color2)
    ax.legend(loc='upper left', frameon=False, fontsize=LEG_FONTSIZE)
    plt.tight_layout()
    plt.show()


# make in_out_corr_mats
label1 = 'tokens betw.\nstart={:,} & end={:,}'.format(START1, END1)
label2 = 'tokens betw.\nstart={:,} & end={:,}'.format(START2, END2)
in_out_corr_mat1, types1 = make_in_out_corr_mat(START1, END1)
in_out_corr_mat2, types2 = make_in_out_corr_mat(START2, END2)

# pca1
print('Fitting PCA 1 ...')
sparse_in_out_corr_mat1 = sparse.csr_matrix(in_out_corr_mat1).asfptype()
u, _, _ = slinalg.svds(sparse_in_out_corr_mat1, k=NUM_PCS)
u1 = u[:, :NUM_PCS]
print(u1.shape)

# pca2
print('Fitting PCA 2 ...')
sparse_in_out_corr_mat2 = sparse.csr_matrix(in_out_corr_mat2).asfptype()
u, _, _ = slinalg.svds(sparse_in_out_corr_mat2, k=NUM_PCS)
u2 = u[:, :NUM_PCS]
print(u2.shape)




# fig for each category
for cat in CATS:
    # y
    cat_probes = [w for w in hub.probe_store.cat_probe_list_dict[cat] if w not in EXCLUDED]
    y1 = calc_measure_at_each_pc(u1, types1, cat_probes, hub.probe_store.types)
    y2 = calc_measure_at_each_pc(u2, types2, cat_probes, hub.probe_store.types)
    # plot
    # plot_comparison(y1, y2, cat, cum=False)
    plot_comparison(y1, y2, cat, cum=True)