import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from childeshub.hub import Hub

DPI = 196
NUM_PARTITIONS = 2
HUB_MODE = 'sem'

hub = Hub(mode=HUB_MODE)

# term distributions
part_size = hub.train_terms.num_tokens // NUM_PARTITIONS
dists = []
fig, ax = plt.subplots(dpi=DPI)
for n, part in enumerate(hub.split(hub.reordered_token_ids, part_size)):
    c = Counter(part)
    dist = np.sort(np.log(list(c.values())))[::-1]
    dists.append(dist)
    ax.plot(dist, label='{}th corpus partition'.format(n + 1))
ax.set_ylabel('Log Freq')
ax.set_xlabel('Term Id')
plt.legend()
plt.show()

