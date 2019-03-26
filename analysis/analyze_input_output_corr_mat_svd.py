import numpy as np
import pyprind
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
import matplotlib.pyplot as plt
from scipy import stats

from childeshub.hub import Hub

HUB_MODE = 'sem'
NUM_PCS = 30
DIST = 1
CORRELATE = False

LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (20, 6)
DPI = 200
YLIM = 10

EXCLUDED = []  # ['may', 'june', 'sweet', 'back', 'behind', 'fly']  # TODO

hub = Hub(mode=HUB_MODE)


CATS = [] or hub.probe_store.cats  #  ['bug', 'dessert', 'body', 'bug']


def make_in_out_corr_mat(start, end):
    print('Making in_out_corr_mat with start={} and end={}'.format(start, end))
    tokens = hub.train_terms.tokens[start:end]
    types = sorted(set(tokens))
    num_types = len(types)
    term2type_id = {t: n for n, t in enumerate(types)}
    # cooc_mat
    cooc_mat = np.zeros((num_types, num_types))
    pbar = pyprind.ProgBar(num_types)
    for row_id, in_term in enumerate(types):
        in_locs = [loc for loc in hub.term_unordered_locs_dict[in_term] if start < loc < (end - DIST)]
        out_locs = np.asarray(in_locs) + DIST
        for out_loc in out_locs:
            out_term = hub.train_terms.tokens[out_loc]
            col_id = term2type_id[out_term]
            cooc_mat[row_id, col_id] += 1  # TODO
            # cooc_mat[row_id, col_id] = 1
        pbar.update()
    # corr_mat

    # TODO
    # res = cosine_similarity(cooc_mat) if CORRELATE else cooc_mat
    res = manhattan_distances(cooc_mat) if CORRELATE else cooc_mat
    return res, types


def calc_t(term_id2term, words, u_col):
    ref = []
    exp = []
    for term_id, val in enumerate(u_col):
        term = term_id2term[term_id]
        if term in words:
            exp.append(val)
        elif term in hub.probe_store.types:
            ref.append(val)
        else:
            pass  # skip non-probes
    # t-statistic
    t, pval = stats.ttest_ind(ref, exp, equal_var=False)
    return abs(t)


def calc_t_at_each_pc(pc_mat, terms, words):
    type_id2term = {n: t for n, t in enumerate(terms)}
    res = []
    for pc_id, u_col in enumerate(pc_mat.T):
        metric = calc_t(type_id2term, words, u_col)
        res.append(metric)
    return res


# make in_out_corr_mats
in_out_corr_mat1, types1 = make_in_out_corr_mat(0, hub.midpoint_loc)
in_out_corr_mat2, types2 = make_in_out_corr_mat(hub.midpoint_loc, hub.train_terms.num_tokens)

# fig for each category
for cat in CATS:
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('Do any of the first {} PCs of cooc mat (dist={})\ncode for the category {}?'.format(
        NUM_PCS, DIST, cat.upper()), fontsize=AX_FONTSIZE)
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
    for label, corr_mat, types in [('partition 1', in_out_corr_mat1, types1),
                                   ('partition 2', in_out_corr_mat2, types2)]:
        # pca
        pca = PCA(n_components=NUM_PCS)  # = u, s, v
        u = pca.fit_transform(corr_mat)  # (num_types, NUM_PCS)
        cat_probes = [w for w in hub.probe_store.cat_probe_list_dict[cat] if w not in EXCLUDED]
        y = calc_t_at_each_pc(u, types, cat_probes)
        #
        ax.plot(y, label=label, linewidth=2)
    ax.legend(loc='best', frameon=False, fontsize=LEG_FONTSIZE,
              bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()