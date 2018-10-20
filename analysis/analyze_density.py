import numpy as np
import matplotlib.pyplot as plt

from childeshub.hub import Hub
from childeshub import config

CORPUS_NAME = 'childes-20180319'
BLOCK_ORDER = 'inc_context-entropy'
ANALYZE_POS = []
HUB_MODE = 'sem'

hub = Hub(mode=HUB_MODE, corpus_name=CORPUS_NAME, block_order=BLOCK_ORDER)

for pos in ANALYZE_POS or sorted(config.Preprocess.pos2tags.keys()):
    # data
    y = []
    part_id = 0
    for tokens in hub.split(hub.reordered_tokens,
                            hub.num_items_in_part):
        try:
            num_in_doc = len([1 for token in tokens if token in getattr(hub, pos + 's')])
        except AttributeError:
            num_in_doc = len([1 for token in tokens if token in hub.probe_store.types])
        y.append(num_in_doc)
        print('Found {} num {}s in part {}'.format(num_in_doc, pos, part_id))
        part_id += 1
    # fig
    _, ax = plt.subplots(dpi=192)
    ax.set_ylabel('Num {}'.format(pos))
    ax.set_xlabel('Partition')
    plt.title(BLOCK_ORDER)
    # plot
    x = np.arange(hub.params.num_parts)
    ax.plot(x, y, '-', alpha=0.5)
    y_fitted = hub.fit_line(x, y)
    ax.plot(x, y_fitted, '-')
    y_rolled = hub.roll_mean(y, 20)
    ax.plot(x, y_rolled, '-')
    plt.show()