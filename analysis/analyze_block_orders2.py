import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import cycle

from childeshub.hub import Hub

CORPUS_NAME = 'childes-20180319'
HUB_MODE = 'sem'
NUM_PARTS = 256
NEARNESS_THR = 0.4
BLOCK_ORDERS = ['dec_punctuation', 'dec_noun', 'dec_noun+punctuation']  # TODO test +

DPI = 192
SMOOTH = 20
PLOT_POS_LIST = ['noun']

hub = Hub(mode=HUB_MODE, num_parts=NUM_PARTS, corpus_name=CORPUS_NAME, part_order='inc_age')
ao_partitions = hub.reordered_partitions

# figs
palette = cycle(sns.color_palette("hls", len(BLOCK_ORDERS)))
for pos in PLOT_POS_LIST:
    # make xys
    xys = []
    for part_order in BLOCK_ORDERS:
        reordered_parts = hub.reorder_parts(part_order)
        y = hub.roll_mean([hub.calc_num_pos_in_part(pos, part) for part in reordered_parts], SMOOTH)
        xys.append((y, part_order))
    # fig
    fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI)
    plt.title('Approximating age-order')
    x1 = np.arange(hub.params.num_parts)
    ax.set_xlabel('Partitions')
    ax.set_ylabel('Number of {}s in partition\n(+linear smoothing)'.format(pos))
    # plot
    y1 = hub.roll_mean([hub.calc_num_pos_in_part(pos, part) for part in ao_partitions], SMOOTH)
    ax.plot(x1, y1, color='black', label='age-ordered')
    for y, bo in xys:
        c = next(palette)
        ax.plot(y, color=c, label=bo)
        if len(BLOCK_ORDERS) == 2:
            ax.fill_between(x1, y1, y, where=None, facecolor=c, interpolate=True, alpha=0.5)
    plt.legend(loc='best', frameon=False)
    plt.show()