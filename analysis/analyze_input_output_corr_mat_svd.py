import numpy as np
import pyprind
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from childeshub.hub import Hub

HUB_MODE = 'sem'
NUM_PCS = 10
DIST = 1
CORRELATE = False
NOUNS = False

LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (20, 6)
DPI = 200

VERBOSE = False
PLOT_BY_CAT = True

hub = Hub(mode=HUB_MODE)


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
            cooc_mat[row_id, col_id] += 1
            # cooc_mat[row_id, col_id] = 1
        pbar.update()
    # corr_mat
    res = cosine_similarity(cooc_mat) if CORRELATE else cooc_mat
    return res, types


def calc_metric(type_id2term, words, u_col):
    """
    this metric is invented by ph: it is the average rank of probes in a category,
    where rank is defined by how high a probe loads compared to all other words in vocab for given PC
    """
    ranks = []
    for rank, type_id in enumerate(np.argsort(u_col)):
        term = type_id2term[type_id]
        if term in words:
            ranks.append(rank)
    return np.mean(ranks)


def calc_metric_at_each_pc(u, types, cat):
    if NOUNS:
        words = hub.nouns
    else:
        words = hub.probe_store.cat_probe_list_dict[cat]
    #
    type_id2term = {n: t for n, t in enumerate(types)}
    res = []
    for pc_id, u_col in enumerate(u.T):
        metric = calc_metric(type_id2term, words, u_col)
        res.append(metric)
    return res


# make in_out_corr_mats
in_out_corr_mat1, types1 = make_in_out_corr_mat(0, hub.midpoint_loc)
in_out_corr_mat2, types2 = make_in_out_corr_mat(hub.midpoint_loc, hub.train_terms.num_tokens)


# fig summarizing across categories
fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
plt.title('Loadings on PCs {}\nderived from word co-occurrences at dist={}'.format(
    'for nouns' if NOUNS else 'across semantic categories', DIST), fontsize=AX_FONTSIZE)
ax.set_ylabel('Average Rank of Category Probes', fontsize=AX_FONTSIZE)
ax.set_xlabel('Principal Component', fontsize=AX_FONTSIZE)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.set_ylim([0, len(hub.train_terms.types)])
ax.axhline(y=len(hub.train_terms.types) / 2, color='grey')
# plot
for label, corr_mat, types in [('partition 1', in_out_corr_mat1, types1),
                               ('partition 2', in_out_corr_mat2, types2)]:
    # pca
    pca = PCA(n_components=NUM_PCS)  # = u, s, v
    u = pca.fit_transform(corr_mat)  # (num_types, NUM_PCS)
    y = np.zeros(NUM_PCS)
    for cat in hub.probe_store.cats:
        y += calc_metric_at_each_pc(u, types, cat)
        if NOUNS:
            break
    #
    ax.plot(y / hub.probe_store.num_cats, label=label, linewidth=2)
ax.legend(loc='best', frameon=False, fontsize=LEG_FONTSIZE,
          bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()

if not PLOT_BY_CAT:
    raise SystemExit

# fig for each category
for cat in hub.probe_store.cats:
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('Do any of the first {} PCs code for the category {}?'.format(NUM_PCS, cat.upper()), fontsize=AX_FONTSIZE)
    ax.set_ylabel('Average Rank of Category Probes'.format(cat.upper()), fontsize=AX_FONTSIZE)
    ax.set_xlabel('Principal Component', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.set_ylim([0, len(hub.train_terms.types)])
    ax.axhline(y=len(hub.train_terms.types) / 2, color='grey')
    # plot
    for label, corr_mat, types in [('partition 1', in_out_corr_mat1, types1),
                                   ('partition 2', in_out_corr_mat2, types2)]:
        # pca
        pca = PCA(n_components=NUM_PCS)  # = u, s, v
        u = pca.fit_transform(corr_mat)  # (num_types, NUM_PCS)
        y = calc_metric_at_each_pc(u, types, cat)
        #
        ax.plot(y, label=label, linewidth=2)
    ax.legend(loc='best', frameon=False, fontsize=LEG_FONTSIZE,
              bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()