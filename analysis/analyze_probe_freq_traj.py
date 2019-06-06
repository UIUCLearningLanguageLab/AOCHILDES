import numpy as np
import matplotlib.pyplot as plt

from childeshub.hub import Hub

NUM_DOCS = 256
DPI = 192
FIGSIZE = (10, 5)
HUB_MODE = 'sem'
BLOCK_ORDER = 'inc_probes-context-entropy-1-right'

hub = Hub(mode=HUB_MODE, part_order=BLOCK_ORDER, num_parts=256)


# data
y = []
for part in hub.reordered_partitions:
    probe_term_ids = [hub.train_terms.term_id_dict[probe] for probe in hub.probe_store.types]
    num_occurences = len([1 for term_id in part if term_id in probe_term_ids])
    print(num_occurences)
    y.append(num_occurences)


# fig
fig, ax = plt.subplots(dpi=DPI, figsize=FIGSIZE)
plt.title(HUB_MODE)
ax.set_ylabel('Num Probe Occurrences in Part')
ax.set_xlabel('Partition')
# plot
x = np.arange(hub.params.num_parts)
ax.plot(x, y, '-', alpha=0.5)
y_fitted = hub.fit_line(x, y)
ax.plot(x, y_fitted, '-')
y_rolled = hub.roll_mean(y, 20)
ax.plot(x, y_rolled, '-')
fig.tight_layout()
plt.show()