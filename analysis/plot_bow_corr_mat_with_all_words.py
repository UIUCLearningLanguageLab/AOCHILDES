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


def cluster(m, dg0, dg1, original_row_words=None, original_col_words=None,
            method='complete', metric='cityblock'):
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
    if original_row_words is None and original_col_words is None:
        return res, dg0, dg1
    else:
        row_labels = np.array(original_row_words)[dg0['leaves']]
        col_labels = np.array(original_col_words)[dg1['leaves']]
        return res, row_labels, col_labels, dg0, dg1


def plot_heatmap(mat, ytick_labels, xtick_labels,
                 figsize=(30, 30), dpi=300, ticklabel_fs=1, title_fs=5):
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
    res = np.clip(np.nan_to_num(np.corrcoef(data_mat, rowvar=True)), -1, 1)  # observations in cols when True
    return res


def get_bow_token_representations(ws_mat, norm=NORM):
    res = np.zeros((hub.params.num_types, hub.params.num_types))
    for window in ws_mat:
        obs_word_id = window[-1]
        for var_word_id in window[:-1]:
            res[obs_word_id, var_word_id] += 1  # TODO which order?
    # norm
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
    token_reps = get_bow_token_representations(windows_mat)
    print('shape of probe_reps={}'.format(token_reps.shape))
    assert len(token_reps) == hub.params.num_types
    # pca
    pca = PCA(n_components=N_COMPONENTS)
    token_reps = pca.fit_transform(token_reps)
    print('shape after PCA={}'.format(token_reps.shape))
    # plot
    corr_mat = to_corr_mat(token_reps)
    print('shape of corr_mat={}'.format(corr_mat.shape))
    clustered_corr_mat, rls, cls, dg0, dg1 = cluster(corr_mat, dg0, dg1, hub.train_terms.types, hub.train_terms.types)
    plot_heatmap(clustered_corr_mat, rls, cls)
    print('------------------------------------------------------')

