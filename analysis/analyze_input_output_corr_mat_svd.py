import numpy as np
import pyprind
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from childeshub.hub import Hub

HUB_MODE = 'sem'
NUM_PCS = 100
CAT = 'animal'
DIST = 1
CORRELATE = False

LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (20, 6)
DPI = 200

VERBOSE = False

hub = Hub(mode=HUB_MODE)

assert CAT in hub.probe_store.cats


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
            # cooc_mat[row_id, col_id] += 1
            cooc_mat[row_id, col_id] = 1
        pbar.update()
    # corr_mat
    res = cosine_similarity(cooc_mat) if CORRELATE else cooc_mat
    return res, types


def calc_loadings_at_each_pc(u, types, cat):
    if cat == 'noun':
        words = hub.nouns
    else:
        words = hub.probe_store.cat_probe_list_dict[cat]
    res = []
    for pc_id, u_col in enumerate(u.T):
        if VERBOSE:
            print('principal component={}'.format(pc_id))
        y_true = np.asarray([1 if t in words else 0 for t in types])
        y_score = u_col
        ap = np.dot(y_true, y_score)
        if VERBOSE:
            print(cat, np.round(ap, 2))
        res.append(ap)
    return res


# make in_out_corr_mats
in_out_corr_mat1, types1 = make_in_out_corr_mat(0, hub.midpoint_loc)
in_out_corr_mat2, types2 = make_in_out_corr_mat(hub.midpoint_loc, hub.train_terms.num_tokens)


for cat in hub.probe_store.cats:
    # fig
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('Do any of the first {} PCs code for the category {}?'.format(NUM_PCS, cat.upper()), fontsize=AX_FONTSIZE)
    ax.set_ylabel('Abs(Sum(Loadings of {} words))'.format(cat.upper()), fontsize=AX_FONTSIZE)
    ax.set_xlabel('Principal Component', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    for label, corr_mat, types in [('partition 1', in_out_corr_mat1, types1),
                                   ('partition 2', in_out_corr_mat2, types2)]:
        # pca
        pca = PCA(n_components=NUM_PCS)  # = u, s, v
        u = pca.fit_transform(corr_mat)  # (num_types, NUM_PCS)
        y = np.abs(calc_loadings_at_each_pc(u, types, cat))
        #
        ax.plot(y, label=label, linewidth=2)
    ax.legend(loc='best', frameon=False, fontsize=LEG_FONTSIZE,
              bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()