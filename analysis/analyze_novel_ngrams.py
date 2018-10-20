import numpy as np
import pyprind
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter

from childeshub.hub import Hub


NUM_BINS = 64
NGRAM_SIZES = [1, 2, 3, 6]
HUB_MODE = 'sem'
BLOCK_ORDERS = ['inc_age', 'dec_age', 'inc_context-entropy', 'dec_context-entropy']

COLORS = ['green', 'orange', 'blue', 'red']
LW = 1
AX_FONTSIZE = 7
LEG_FONTSIZE = 6
DPI = 192
IS_LOG = True
WSPACE = 0.0
HSPACE = 0.0
WPAD = 0.0
HPAD = 0.0
PAD = 0.2


def human_format(num, pos):  # pos is required
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format(num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def make_novel_xys(ngrams_list, num_bins=NUM_BINS):
    """
    Return tuples containing x and y data for plotting trajs of novel term occurrences in "corpora"
    """
    print('Making novel_xy...')
    num_ngram_lists = len(ngrams_list)
    num_ngrams = len(ngrams_list[0])
    print('Number of n-grams: {:,}'.format(num_ngrams))
    assert len(set([len(c) for c in ngrams_list])) == 1
    trajs = [np.zeros(num_ngrams) for _ in range(num_ngram_lists)]
    seen_sets = [set() for _ in range(num_ngram_lists)]
    pbar = pyprind.ProgBar(num_ngrams)
    # trajs
    for n, (zipped_terms) in enumerate(zip(*ngrams_list)):
        pbar.update()
        for zip_id, term in enumerate(zipped_terms):
            if term not in seen_sets[zip_id]:
                trajs[zip_id][n] = 1
                seen_sets[zip_id].add(term)
            else:
                trajs[zip_id][n] = np.nan
    # xys
    result = []
    for traj in trajs:
        ns = np.where(traj == 1.00)[0]
        hist, b = np.histogram(ns, bins=num_bins, range=[0, num_ngrams])
        xy = (b[:-1], hist)
        result.append(xy)
    return result


hub = Hub(mode=HUB_MODE)
num_ngram_sizes = len(NGRAM_SIZES)

# make size_ngrams_list_dict
size_ngrams_list_dict = {ngram_size: [] for ngram_size in NGRAM_SIZES}
for part_order in BLOCK_ORDERS:
    hub.part_order = part_order
    for ngram_size in NGRAM_SIZES:
        ngram_range = (ngram_size, ngram_size)
        ngrams = hub.get_ngrams(ngram_range, hub.reordered_tokens)
        size_ngrams_list_dict[ngram_size].append(ngrams)
    del hub.__dict__['reordered_partitions']
    del hub.__dict__['reordered_token_ids']
    del hub.__dict__['reordered_tokens']

# make xys
xys_list = []
for ngram_size in NGRAM_SIZES:
    ngrams_list = size_ngrams_list_dict[ngram_size]  # ngrams_list contains ngrams for each part_order
    xys = make_novel_xys(ngrams_list)
    xys_list.append(xys)

# fig
fig, axs = plt.subplots(num_ngram_sizes, 1, sharex='all', dpi=DPI, figsize=(8, 0.8 * num_ngram_sizes))
if num_ngram_sizes == 1:
    axs = [axs]
for ax, xys, ngram_size in zip(axs, xys_list, NGRAM_SIZES):
    if ax == axs[-1]:
        ax.set_xlabel('Number of n-grams', fontsize=AX_FONTSIZE)
        ax.tick_params(axis='both', which='both', top='off', right='off')
    else:
        ax.tick_params(axis='both', which='both', top='off', right='off', bottom='off')
    # if ax == axs[-num_ngram_sizes // 2]:
    #     ax.set_ylabel('Log Frequency of Novel n-grams'.format(ngram_size), fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    plt.setp(ax.get_yticklabels(), fontsize=LEG_FONTSIZE)
    plt.setp(ax.get_xticklabels(), fontsize=LEG_FONTSIZE)
    # ax.set_title('{}-gram'.format(ngram_size), fontsize=LEG_FONTSIZE, y=0.9)
    ax.set_ylabel('Log F Novel\n{}-grams'.format(ngram_size), fontsize=AX_FONTSIZE)
    ax.yaxis.grid(True)
    # plot
    for (x, y), c, label in zip(xys, COLORS, BLOCK_ORDERS):
        if IS_LOG:
            y = np.log(np.clip(y, 1, np.max(y)))
        ax.plot(x, y, c=c, linewidth=LW, label=label, linestyle='-')
# show
plt.legend(loc='upper center', fontsize=LEG_FONTSIZE, frameon=False,
               bbox_to_anchor=(0.5, 0.3), ncol=3)
plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
plt.tight_layout(h_pad=HPAD, w_pad=WPAD, pad=PAD)
plt.show()


# blow up
_, ax = plt.subplots(dpi=DPI, figsize=(6.0, 3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top='off', right='off')
plt.setp(ax.get_yticklabels(), fontsize=LEG_FONTSIZE)
plt.setp(ax.get_xticklabels(), fontsize=LEG_FONTSIZE)
ax.set_xlim([0, 10000])
ax.set_ylim([6, 8])
ax.xaxis.set_major_formatter(FuncFormatter(human_format))
# plot
for (x, y), c, label in zip(xys_list[0], COLORS, BLOCK_ORDERS):
    print(label)
    print('num novel 1-grams:', y[:2])
    print()
    if IS_LOG:
        y = np.log(np.clip(y, 1, np.max(y)))
    ax.plot(x, y, c=c, linewidth=LW, label=label, linestyle='-')
# show
plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
plt.tight_layout(h_pad=HPAD, w_pad=WPAD, pad=PAD)
plt.show()