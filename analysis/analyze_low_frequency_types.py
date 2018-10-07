import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hub import Hub

HUB_MODE = 'sem'
CORPUS_NAME = 'childes-20180319'  # TODO
# CORPUS_NAME = 'childes-20171212'  # TODO
BLOCK_ORDERS = ['inc_age', 'dec_age']
NUM_PARTS = 2
NUM_TYPES = 4096 * 1  # num types in raw: 26,639  # min_freq when 4096 is 28
FREQUENCIES = [10 * 1000]  # 10,000 is good upper freq bound for content words

title = '# types with corpus frequency < min_freq + f'


def calc_y(hub, freq):
    print('Calculating y with f={}...'.format(freq))
    min_freq_in_corpus = sorted(hub.train_terms.term_freq_dict.items(), key=lambda i: i[1])[0][1]
    result = []
    for part_id, part in enumerate(hub.reordered_partitions):
        terms = [hub.train_terms.types[term_id] for term_id in part]
        freqs = [hub.train_terms.term_freq_dict[term] for term in terms]
        num_filtered_types = np.sum([1 for f in freqs if min_freq_in_corpus + freq >= f >= min_freq_in_corpus])  # TODO
        print([(t, f) for t, f in zip(terms, freqs) if min_freq_in_corpus + freq >= f >= min_freq_in_corpus][:10])  # TODO
        result.append(num_filtered_types)
    return result




# fig
fig, ax = plt.subplots(dpi=192)
plt.title('{} (num_types={:,})'.format(CORPUS_NAME, NUM_TYPES))
ax.set_xlabel('Partition')
ax.set_ylabel(title)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top='off', right='off')
ax.yaxis.grid(True)
colors = sns.color_palette("hls", len(BLOCK_ORDERS))
for c, bo in zip(colors, BLOCK_ORDERS):
    hub = Hub(mode=HUB_MODE, num_types=NUM_TYPES, corpus_name=CORPUS_NAME, num_parts=NUM_PARTS, block_order=bo)
    # for k, v in sorted(hub.train_terms.term_freq_dict.items(), key=lambda i: i[1]):
    #     print(k, v)
    for f in FREQUENCIES:
        y = calc_y(hub, f)
        ax.plot(y, '-', label='f={:,}, {}'.format(f, bo), color=c)
plt.legend()
plt.show()






