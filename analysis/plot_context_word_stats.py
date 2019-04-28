import numpy as np
import matplotlib.pyplot as plt
import pyprind

from childeshub.hub import Hub

NGRAM_SIZE = 1

FIGSIZE = (6, 6)
TITLE_FONTSIZE = 10
COLORS = ['red', 'blue']

HUB_MODE = 'sem'


hub = Hub(mode=HUB_MODE)
cats = hub.probe_store.cats
probe2cat = hub.probe_store.probe_cat_dict
vocab = hub.train_terms.types


def make_ngram_count_mat(contexts_mat, num_vocab):
    #
    pbar = pyprind.ProgBar(len(contexts_mat))
    res = np.zeros((num_contexts, num_vocab))
    for context, col_id in zip(contexts_mat, outputs):
        row_id = context2id[tuple(context)]
        res[row_id, col_id] += 1
        pbar.update()
    return res


part_ids = range(2)
cat2part_id2y1 = {cat: {part_id: None for part_id in part_ids} for cat in cats}
cat2part_id2y2 = {cat: {part_id: None for part_id in part_ids} for cat in cats}
for part_id in part_ids:
    windows_mat = hub.make_windows_mat(hub.reordered_partitions[part_id], hub.num_windows_in_part)
    #
    assert windows_mat.shape[1] > NGRAM_SIZE  # last word is not part of context
    print('Making In-Out matrix...')
    inputs_mat, outputs = windows_mat[:, :-1], windows_mat[:, -1]
    contexts_mat = inputs_mat[:, -NGRAM_SIZE:]
    # count_mat
    unique_contexts = np.unique(contexts_mat, axis=0)
    num_contexts = len(unique_contexts)
    context2id = {tuple(c): n for n, c in enumerate(unique_contexts)}
    count_mat = make_ngram_count_mat(contexts_mat, hub.params.num_types)

    # plot statistic not collapsing over categories
    for cat_id, cat in enumerate(cats):
        cat_probes = [p for p in hub.probe_store.types if probe2cat[p] == cat]
        term_ids = [hub.train_terms.term_id_dict[p] for p in cat_probes]
        filtered_rows = [counts for context_id, counts in enumerate(count_mat)
                         if unique_contexts[context_id][-1] in term_ids]
        #
        y1 = np.sort([row.var() for row in filtered_rows])
        y2 = np.sort([np.count_nonzero(row) for row in filtered_rows])
        cat2part_id2y1[cat][part_id] = y1
        cat2part_id2y2[cat][part_id] = y2
    print('------------------------------------------------------')


for cat, part_id2ys in cat2part_id2y1.items():
    # fig1
    fig1, ax1 = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title('CHILDES {}-gram Context Statistics\n{}'.format(NGRAM_SIZE, cat.upper()),
              fontsize=TITLE_FONTSIZE)
    ax1.set_xlabel('Category Member')
    ax1.set_ylabel('Variance of context word counts')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(axis='both', which='both', top=False, right=False)
    #
    for part_id, y in part_id2ys.items():
        label = 'partition={}'.format(part_id)
        ax1.plot(y,
                 label=label,
                 color=COLORS[part_id])
    ax1.legend(frameon=False)
    plt.tight_layout()
    plt.show()

# fig2
for cat, part_id2ys in cat2part_id2y2.items():
    fig2, ax2 = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title('CHILDES {}-gram Context Statistics\n{}'.format(NGRAM_SIZE, cat.upper()),
              fontsize=TITLE_FONTSIZE)
    ax2.set_xlabel('Category Member')
    ax2.set_ylabel('Number of context word types')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='both', which='both', top=False, right=False)
    #
    for part_id, y in part_id2ys.items():
        label = 'partition={}'.format(part_id)
        ax2.plot(y,
                 label=label,
                 color=COLORS[part_id])
    ax2.legend(frameon=False)
    plt.tight_layout()
    plt.show()