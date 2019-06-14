import matplotlib.pyplot as plt
from scipy.sparse import linalg as slinalg
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

from childeshub.hub import Hub


HUB_MODE = 'sem'
WINDOW_SIZE = 3
NUM_SVS = 64


def plot_comparison(y1, y2, fontsize=16):
    fig, ax = plt.subplots(1, figsize=(5, 5), dpi=None)
    plt.title('SVD of AO-CHILDES partitions'
              '\n(window size={})'.format(WINDOW_SIZE), fontsize=fontsize)
    ax.set_ylabel('singular value', fontsize=fontsize)
    ax.set_xlabel('Singular Dimension', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    ax.plot(y1, label=label1, linewidth=2)
    ax.plot(y2, label=label2, linewidth=2)
    ax.legend(loc='upper right', frameon=False, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


hub = Hub(mode=HUB_MODE)

# make term_by_window_co_occurrence_mats
start1, end1 = 0, hub.midpoint_loc // 1
start2, end2 = hub.train_terms.num_tokens - end1, hub.train_terms.num_tokens
label1 = 'partition 1' or 'tokens between\n{:,} & {:,}'.format(start1, end1)
label2 = 'partition 2' or 'tokens between\n{:,} & {:,}'.format(start2, end2)
tw_mat1, xws1, yws1 = hub.make_term_by_window_co_occurrence_mat(start=start1, end=end1, window_size=WINDOW_SIZE)
tw_mat2, xws2, yws2 = hub.make_term_by_window_co_occurrence_mat(start=start2, end=end2, window_size=WINDOW_SIZE)


# analyze s
singular_vals1 = []
singular_vals2 = []
for y, mat in [(singular_vals1, tw_mat1.asfptype()),
               (singular_vals2, tw_mat2.asfptype())]:
    # compute variance of sparse matrix
    scaler = StandardScaler(with_mean=False).fit(mat)
    print('sum of column variances of term-by-window co-occurrence matrix={:,}'.format(scaler.var_.sum()))
    # SVD
    print('Fitting SVD ...')
    normalized = normalize(mat, axis=1, norm='l2', copy=False)
    _, s, _ = slinalg.svds(normalized, k=NUM_SVS, return_singular_vectors='vh')  # s is not 2D
    print('sum of singular values={:,}'.format(s.sum()))
    print('var of singular values={:,}'.format(s.var()))
    # collect singular values
    for sing_val in s[:-1][::-1]:  # last s is combination of all remaining s
        y.append(sing_val)
    print()

plot_comparison(singular_vals1, singular_vals2)