import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize
from itertools import chain

from hub import Hub

HUB_MODE = 'sem'

VERBOSE = False
SHOW_PLOTS = False
MAX_NUM_CONTEXTS = 50  # number of most frequent contexts for each category
CONTEXT_SIZE = 4  # number of words on either side of probe
NORMALIZATION = 'l2'  # l1, l2, max
NUM_PROTOTYPE_CONTEXTS = 5  # proportion of same-category contexts a probe must occur with to be determined a prototype
RELATIVE_FREQUENCY = True  # choose categories based on freq if False, otherwise based on cat_freq > total - cat_freq

hub = Hub(mode=HUB_MODE)
num_rows = hub.probe_store.num_cats * MAX_NUM_CONTEXTS
probes_sorted_by_cat = sorted(hub.probe_store.types, key=hub.probe_store.probe_cat_dict.get)  # to sort x-axis by cat


def make_contexts_dicts(tokens):  # bidirectional context
    cat_contexts_d = {cat: [] for cat in hub.probe_store.cats}
    probe_context_d = {probe: [] for probe in hub.probe_store.types}
    for n, t in enumerate(tokens):
        if t in hub.probe_store.types:
            context = tuple(tokens[n - CONTEXT_SIZE:n] +
                            tokens[n + 1: n + 1 + CONTEXT_SIZE])
            cat = hub.probe_store.probe_cat_dict[t]
            cat_contexts_d[cat].append(context)
            probe_context_d[t].append(context)
    return cat_contexts_d, probe_context_d


def make_cat_competition_mat(tokens):
    # choose contexts
    print('Making context dicts...')
    cat_contexts_dict, probe_contexts_dict = make_contexts_dicts(tokens)
    if RELATIVE_FREQUENCY:
        num = 0
        cat_contexts_d = {}
        total_counter = Counter(list(chain(*[contexts for contexts in cat_contexts_dict.values()])))
        for cat, cs in cat_contexts_dict.items():
            cat_counter = Counter(cs)
            comparatively_most_freq_contexts = [c for c in set(cs) if
                                                cat_counter[c] > total_counter[c] - cat_counter[c]]
            if VERBOSE:
                print('{:<12} : {:>6}'.format(cat, len(comparatively_most_freq_contexts)))
            for c in comparatively_most_freq_contexts:
                if VERBOSE:
                    print(c, cat_counter[c])
                num += 1
            comparatively_most_freq_contexts = sorted(comparatively_most_freq_contexts,
                                                      key=cat_counter.get)[::-1][:MAX_NUM_CONTEXTS]
            cat_contexts_d[cat] = comparatively_most_freq_contexts
        print('Total number of contexts occurring more frequently with category members than non-members: {}'.format(num))
    else:
        cat_contexts_d = {}
        for cat, cs in cat_contexts_dict.items():
            most_freq_contexts = [i[0] for i in Counter(cs).most_common(MAX_NUM_CONTEXTS)]
            if VERBOSE:
                print(cat, most_freq_contexts)
            cat_contexts_d[cat] = most_freq_contexts
    # make category competition matrix (iterate over most_freq_contexts for each cat [rows], and over probes [cols])
    print('Making heatmap...')
    global contexts
    contexts = []
    mat = np.zeros((num_rows, hub.probe_store.num_probes), dtype=np.int)
    for i in range(num_rows):
        cat_id = i // MAX_NUM_CONTEXTS
        context_id = i % MAX_NUM_CONTEXTS
        cat = hub.probe_store.cats[cat_id]
        context = cat_contexts_d[cat][context_id]
        contexts.append(' '.join(context[:CONTEXT_SIZE]) + ' _ ' + ' '.join(context[CONTEXT_SIZE:]))  # for fig labels
        for j, probe in enumerate(probes_sorted_by_cat):
            probe_contexts = probe_contexts_dict[probe]
            num_occurences = probe_contexts.count(context)  # count number of times probe occurs in cat context
            mat[i, j] = num_occurences
            if VERBOSE:
                print('{:<45} {:<12} {:<4}'.format(str(context), probe, num_occurences))
    return mat


def plot_clustermap(m, part_id, col_ids=None, row_ids=None, cluster_cols=False, cluster_rows=False):
    fig, ax_heatmap = plt.subplots(figsize=(20, 10))
    title = '{} Category Context Competition'.format('Syntactic' if HUB_MODE == 'syn' else 'Semantic')
    title += '\nPartition {}'.format(part_id)
    divider = make_axes_locatable(ax_heatmap)
    ax_dendleft = divider.append_axes("right", 1.0, pad=0.0, sharey=ax_heatmap)
    ax_dendleft.set_frame_on(False)
    ax_dendtop = divider.append_axes("top", 1.0, pad=0.0, sharex=ax_heatmap)
    ax_dendtop.set_frame_on(False)
    # side dendrogram
    if cluster_rows:
        lnk0 = linkage(pdist(m))
        dg0 = dendrogram(lnk0,
                         ax=ax_dendleft,
                         orientation='right',
                         color_threshold=-1,
                         no_labels=True)
        z = m[dg0['leaves'], :]  # reorder rows
        z = z[::-1]  # reverse to match orientation of dendrogram
    else:
        z = m
    # top dendrogram
    if cluster_cols:
        lnk1 = linkage(pdist(m.T))
        dg1 = dendrogram(lnk1,
                         ax=ax_dendtop,
                         color_threshold=-1,
                         no_labels=True)
        z = z[:, dg1['leaves']]  # reorder cols to match leaves of dendrogram
        z = z[::-1]  # reverse to match orientation of dendrogram
    else:
        z = z
    # heatmap
    max_x_extent = ax_dendtop.get_xlim()[1]
    max_y_extent = ax_dendleft.get_ylim()[1]
    ax_heatmap.imshow(z,
                      aspect='auto',
                      cmap=plt.cm.jet,
                      interpolation='nearest',
                      extent=(0, max_x_extent, 0, max_y_extent))
    # label axes
    ax_heatmap.set_ylabel('bidirectional category contexts\n(size={}, {} most freq. / category)'.format(
        CONTEXT_SIZE, MAX_NUM_CONTEXTS), fontsize=24)
    if col_ids is not None:
        num_cols = len(col_ids)
        assert num_cols == m.shape[1]
        halfxw = 0.5 * max_x_extent / num_cols
        ax_heatmap.set_xticks(np.linspace(halfxw, max_x_extent - halfxw, num_cols))
        if cluster_cols:
            reordered_labels = np.array([hub.probe_store.types[i] for i in col_ids])[dg1['leaves']][::-1]
        else:
            reordered_labels = np.array([hub.probe_store.types[i] for i in col_ids])
        ax_heatmap.xaxis.set_ticklabels(reordered_labels, rotation=90, fontsize=18)
        ax_heatmap.yaxis.set_ticklabels([])
        title += ' ({} columns above threshold={})'.format(num_cols, NUM_PROTOTYPE_CONTEXTS)
        ax_heatmap.set_xlabel('Probes', fontsize=24)
    elif row_ids is not None:
        num_rows = len(row_ids)
        assert num_rows == m.shape[0]
        halfyw = 0.5 * max_y_extent / num_rows
        ax_heatmap.set_yticks(np.linspace(halfyw, max_y_extent - halfyw, num_rows))
        ax_heatmap.xaxis.set_ticklabels([])
        if cluster_rows:
            reordered_labels = np.array([contexts[i] for i in row_ids])[dg0['leaves']][::-1]
        else:
            reordered_labels = np.array([contexts[i] for i in row_ids])
        ax_heatmap.yaxis.set_ticklabels(reordered_labels, rotation=0)
        title += ' ({} rows above threshold={})'.format(num_rows, NUM_PROTOTYPE_CONTEXTS)
        ax_heatmap.set_xlabel('Probes (category-sorted)', fontsize=24)
    else:
        ax_heatmap.xaxis.set_ticklabels([])
        ax_heatmap.yaxis.set_ticklabels([])
        ax_heatmap.set_xlabel('Probes', fontsize=24)
    plt.title(title, fontsize=30)
    # remove dendrogram ticklines
    lines = (ax_dendtop.xaxis.get_ticklines() +
             ax_dendtop.yaxis.get_ticklines() +
             ax_dendleft.xaxis.get_ticklines() +
             ax_dendleft.yaxis.get_ticklines())
    plt.setp(lines, visible=False)
    # make dendrogram labels invisible
    plt.setp(ax_dendleft.get_yticklabels() + ax_dendleft.get_xticklabels(),
             visible=False)
    plt.setp(ax_dendtop.get_xticklabels() + ax_dendtop.get_yticklabels(),
             visible=False)
    fig.tight_layout()
    plt.show()


mat1 = make_cat_competition_mat(hub.first_half_tokens)
mat2 = make_cat_competition_mat(hub.second_half_tokens)

prototypes_lists = []
for part_id, mat_not_normalized in enumerate([mat1, mat2]):

    mat = normalize(mat_not_normalized, axis=1, norm=NORMALIZATION)  # normalize to avoid frequency effects of probes

    # show all cols
    if SHOW_PLOTS:
        plot_clustermap(mat, part_id)

    # show mat filtered by cols with most nonzero values in filtered locations (where category aligns with probe)
    filtered = mat_not_normalized.copy()
    num_non_matching = 0
    for i in range(num_rows):
        cat_id = i // MAX_NUM_CONTEXTS
        context_id = i % MAX_NUM_CONTEXTS
        cat = hub.probe_store.cats[cat_id]
        no_match_ids = [n for n, p in enumerate(probes_sorted_by_cat)
                        if hub.probe_store.probe_cat_dict[p] != cat]
        num_non_matching += np.sum(mat_not_normalized[i, no_match_ids])
        filtered[i, no_match_ids] = 0

    # show mat filtered by cols with sum above THRESHOLD
    if SHOW_PLOTS:
        col_ids = np.where(np.count_nonzero(filtered, axis=0) > NUM_PROTOTYPE_CONTEXTS)[0]
        col_reduced = mat[:, col_ids]
        plot_clustermap(col_reduced, part_id, col_ids=col_ids, cluster_cols=True)

    # show mat filtered by rows with sum above THRESHOLD
    if SHOW_PLOTS:
        row_ids = np.where(np.count_nonzero(filtered, axis=1) > NUM_PROTOTYPE_CONTEXTS)[0]
        row_reduced = mat[row_ids, :]
        plot_clustermap(row_reduced, part_id, row_ids=row_ids, cluster_rows=True)

    # collect
    if SHOW_PLOTS:
        prototypes = set([hub.probe_store.types[i] for i in col_ids])
        prototypes_lists.append(prototypes)

    # measure interference
    num_total = np.sum(mat_not_normalized)
    print('===========')
    print('# non-matching category-context occurrences={:,}'.format(num_non_matching))
    print('#        total category-context occurrences={:,}'.format(num_total))
    print('# non-matching / # total co-occurrences = {} ("category-competition")'.format(
        (num_non_matching + 1) / (num_total + 1)))
    print('#     matching / # total co-occurrences = {} ("category-cooperation")'.format(
        (num_total - num_non_matching + 1) / (num_total + 1)))
    print('#     matching / # non-matching = {} ("r")'.format(
        (num_total - num_non_matching + 1) / (num_non_matching + 1)))
    print('===========')


# compare partitions
if SHOW_PLOTS:
    protos1, protos2 = prototypes_lists
    num_overlap_prototypes = len([p for p in protos2 if p in protos1])
    for cat in hub.probe_store.cats:
        cat_prototypes1 = sorted([p for p in hub.probe_store.cat_probe_list_dict[cat] if p in protos1])
        cat_prototypes2 = sorted([p for p in hub.probe_store.cat_probe_list_dict[cat] if p in protos2])
        info = '{} -> {}'.format(cat_prototypes1, cat_prototypes2)
        print('{:<12} prototypes: {}'.format(cat, info if cat_prototypes1 or cat_prototypes2 else ''))


# TODO hypothesis: balAcc (when age-reversed) is worse for categories where prototypes change over time
# TODO and balAcc difference (between age conditions) should be minimal when protoytpes are symmetric
