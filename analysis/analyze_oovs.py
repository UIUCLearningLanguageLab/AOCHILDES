import numpy as np
import matplotlib.pyplot as plt

from childeshub.hub import Hub
from childeshub import config

HUB_MODE = 'sem'
NUM_PARTS = 8
NUM_TYPES = 4096 * 1 # num types in raw: 26,639  # TODO training is not over raw corpus
CORPUS_NAME = 'childes-20180319'  # TODO
# CORPUS_NAME = 'childes-20171212'  # TODO


def calc_y():
    result = []
    for part in hub.reordered_partitions:
        terms = [hub.train_terms.types[term_id] for term_id in part]
        num_oovs = np.sum([1 if term == config.Terms.OOV_SYMBOL else 0 for term in terms])
        result.append(num_oovs)
    return result


hub = Hub(mode=HUB_MODE, num_types=NUM_TYPES, corpus_name=CORPUS_NAME, num_parts=NUM_PARTS)

# fig
fig, ax = plt.subplots(dpi=192)
plt.title('{} (num_types={:,})'.format(CORPUS_NAME, NUM_TYPES))
ax.set_xlabel('Partition')
ax.set_ylabel('Number of OOVs')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top='off', right='off')
ax.yaxis.grid(True)
y = calc_y()
ax.plot(y, '-')
plt.show()






