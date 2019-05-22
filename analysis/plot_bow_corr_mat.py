import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA

from childeshub.hub import Hub


"""
plot correlation matrix of BOW model representations - does correlation matrix look more hierarchical in partition 2?
"""

HUB_MODE = 'sem'
BPTT_STEPS = 3  # 3
DIRECTION = -1  # context is left if -1, context is right if +1  # -1
N_COMPONENTS = 512  # 512
NORM = 'l1'  # l1
PART_IDS = [0, 1]  # this is useful because clustering of second corr_mat is based on dg0 and dg1 of first


def cluster(m, dg0, dg1, method='single', metric='cityblock'):
    print('Clustering...')
    #
    if dg0 is None:
        lnk0 = linkage(m, method=method, metric=metric)
        dg0 = dendrogram(lnk0,
                         ax=None,
                         color_threshold=None,
                         no_labels=True,
                         no_plot=True)
    res = m[dg0['leaves'], :]  # reorder rows
    #
    if dg1 is None:
        lnk1 = linkage(m.T, method=method, metric=metric)
        dg1 = dendrogram(lnk1,
                         ax=None,
                         color_threshold=None,
                         no_labels=True,
                         no_plot=True)
    #
    res = res[:, dg1['leaves']]  # reorder cols
    return res, dg0, dg1


def plot_heatmap(mat, ytick_labels, xtick_labels,
                 figsize=(10, 10), dpi=None, ticklabel_fs=1, title_fs=5):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.title('', fontsize=title_fs)
    # heatmap
    print('Plotting heatmap...')
    ax.imshow(mat,
              aspect='equal',
              cmap=plt.get_cmap('jet'),
              interpolation='nearest')
    # xticks
    num_cols = len(mat.T)
    ax.set_xticks(np.arange(num_cols))
    ax.xaxis.set_ticklabels(xtick_labels, rotation=90, fontsize=ticklabel_fs)
    # yticks
    num_rows = len(mat)
    ax.set_yticks(np.arange(num_rows))
    ax.yaxis.set_ticklabels(ytick_labels,   # no need to reverse (because no extent is set)
                            rotation=0, fontsize=ticklabel_fs)
    # remove ticklines
    lines = (ax.xaxis.get_ticklines() +
             ax.yaxis.get_ticklines())
    plt.setp(lines, visible=False)
    plt.show()


def to_corr_mat(data_mat):
    mns = data_mat.mean(axis=1, keepdims=True)
    stds = data_mat.std(axis=1, ddof=1, keepdims=True) + 1e-6  # prevent np.inf (happens when dividing by zero)
    deviated = data_mat - mns
    zscored = deviated / stds
    res = np.matmul(zscored, zscored.T) / len(data_mat) # it matters which matrix is transposed
    return res


def get_bow_probe_representations(ws_mat, norm=NORM):
    probe2rep = {p: np.zeros(hub.params.num_types) for p in hub.probe_store.types}
    for window in ws_mat:
        first_word = hub.train_terms.types[window[0]]
        last_word = hub.train_terms.types[window[-1]]
        if DIRECTION == -1:  # context is defined to be words left of probe
            if last_word in hub.probe_store.types:
                for word_id in window[:-1]:
                    probe2rep[last_word][word_id] += 1
        elif DIRECTION == 1:
            if first_word in hub.probe_store.types:
                for word_id in window[0:]:
                    probe2rep[first_word][word_id] += 1
        else:
            raise AttributeError('Invalid arg to "DIRECTION".')
    # representations
    res = np.asarray([probe2rep[p] for p in hub.probe_store.types])
    if norm is not None:
        res = normalize(res, axis=1, norm=norm, copy=False)
    return res


hub = Hub(mode=HUB_MODE, bptt_steps=BPTT_STEPS)
cats = hub.probe_store.cats
probe2cat = hub.probe_store.probe_cat_dict
vocab = hub.train_terms.types


#
dg0, dg1 = None, None
for part_id in PART_IDS:
    # a window is [x1, x2, x3, x4, x5, x6, x7, y] if bptt=7
    windows_mat = hub.make_windows_mat(hub.reordered_partitions[part_id], hub.num_windows_in_part)
    print('shape of windows_mat={}'.format(windows_mat.shape))
    probe_reps = get_bow_probe_representations(windows_mat)
    print('shape of probe_reps={}'.format(probe_reps.shape))
    assert len(probe_reps) == hub.probe_store.num_probes
    # transform representations
    pca = PCA(n_components=N_COMPONENTS)
    probe_reps = pca.fit_transform(probe_reps)
    print('shape after PCA={}'.format(probe_reps.shape))
    probe_reps = to_corr_mat(probe_reps)
    # plot
    print('shape of heatmap={}'.format(probe_reps.shape))
    clustered_corr_mat, dg0, dg1 = cluster(probe_reps, dg0, dg1)
    plot_heatmap(clustered_corr_mat, [], [])
    print('------------------------------------------------------')

