import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy import optimize

from childeshub.hub import Hub

HUB_MODE = 'sem'
NUM_PARTS = 4
NUM_TYPES = 1000 * 26
SPLIT_SIZE = 5620
CORPUS_ID = 1  # 0=mobydick, 1=CHILDES


def fitfunc(p, x):
    return p[0] + p[1] * x


def errfunc(p, x, y):
    return y - fitfunc(p, x)


corpus_name = ['mobydick', 'childes-20180319'][CORPUS_ID]
hub = Hub(mode=HUB_MODE, num_types=NUM_TYPES, corpus_name=corpus_name, num_parts=NUM_PARTS)


for part_id, part in enumerate(hub.reordered_partitions):
    # make freq_mat
    num_splits = hub.num_items_in_part // SPLIT_SIZE + 1
    freq_mat = np.zeros((hub.train_terms.num_types, num_splits))
    start_locs = np.arange(0, hub.num_items_in_part, SPLIT_SIZE)
    num_start_locs = len(start_locs)
    for split_id, start_loc in enumerate(start_locs):
        for token_id, f in Counter(part[start_loc:start_loc + SPLIT_SIZE]).items():
            freq_mat[token_id, split_id] = f
    # x, y
    freq_mat = freq_mat[~np.all(freq_mat == 0, axis=1)]
    x = freq_mat.mean(axis=1)  # make sure not to have rows with zeros
    y = freq_mat.std(axis=1)
    # fit
    pinit = np.array([1.0, -1.0])
    logx = np.log10(x)
    logy = np.log10(y)
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=True)

    for i in out:
        print(i)

    pfinal = out[0]
    amp = pfinal[0]
    alpha = pfinal[1]
    # print(part_id, alpha)
    # fig
    fig, ax = plt.subplots()
    plt.title('{} (num_types={:,}, part# {} of{})'.format(
        corpus_name, NUM_TYPES, part_id + 1, NUM_PARTS))
    ax.set_xlabel('mean')
    ax.set_ylabel('std')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.text(x=1.0, y=0.3, s='Taylor\'s exponent: {:.3f}'.format(alpha))
    ax.loglog(x, y, '.', markersize=2)

    # TODO
    ax.loglog(x, amp * (x ** alpha) + 0, '.', markersize=2)


    plt.show()






