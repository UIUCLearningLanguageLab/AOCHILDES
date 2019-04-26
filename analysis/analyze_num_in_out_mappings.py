import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from childeshub.hub import Hub

WINDOW_SIZES = [2, 3, 4, 5, 6]

hub = Hub(mode='sem')
windows_mat1 = hub.make_windows_mat(hub.reordered_partitions[0], hub.num_windows_in_part)
windows_mat2 = hub.make_windows_mat(hub.reordered_partitions[1], hub.num_windows_in_part)


def calc_y(w_mat, w_size, uniq):
    truncated_w_mat = w_mat[:, -w_size:]
    u = np.unique(truncated_w_mat, axis=0)
    num_total_windows = len(truncated_w_mat)
    num_uniq = len(u)
    num_repeated = num_total_windows - num_uniq
    #
    print(num_total_windows, num_uniq, num_repeated)
    if uniq:
        return len(u)
    else:
        return num_repeated


def plot(ylabel, ys_list):
    bar_width0 = 0.0
    bar_width1 = 0.25
    _, ax = plt.subplots(dpi=192)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('punctuation')
    ax.set_xlabel('window_size')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    num_conditions = len(WINDOW_SIZES)
    xs = np.arange(1, num_conditions + 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(WINDOW_SIZES)
    # plot
    colors = sns.color_palette("hls", 2)[::-1]
    labels = ['partition 1', 'partition 2']
    for n, (x, ys) in enumerate(zip(xs, ys_list)):
        ax.bar(x + bar_width0, ys[0], bar_width1, color=colors[0], label=labels[0] if n == 0 else '_nolegend_')
        ax.bar(x + bar_width1, ys[1], bar_width1, color=colors[1], label=labels[1] if n == 0 else '_nolegend_')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


plot('Number of repeated IO Mappings', [(calc_y(windows_mat1, ws, False), calc_y(windows_mat2, ws, False))
                                        for ws in WINDOW_SIZES])

plot('Number of unique IO Mappings', [(calc_y(windows_mat1, ws, True), calc_y(windows_mat2, ws, True))
                                      for ws in WINDOW_SIZES])