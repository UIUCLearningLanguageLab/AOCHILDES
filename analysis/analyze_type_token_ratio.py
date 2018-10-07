import matplotlib.pyplot as plt
import numpy as np

from hub import Hub

HUB_MODE = 'sem'
BLOCK_ORDER = 'inc_age'
DPI = 192
NUM_PARTS = 2
IS_REORDERED = False  # true if lateness of probes is determined after partition reordering

hub = Hub(mode=HUB_MODE, num_parts=NUM_PARTS, block_order=BLOCK_ORDER)
probes_early, probes_late = hub.split_probes_by_loc(2, is_reordered=IS_REORDERED)  # typically unordered
probe_term_ids_early = [hub.train_terms.term_id_dict[probe] for probe in probes_early]
probe_term_ids_late = [hub.train_terms.term_id_dict[probe] for probe in probes_late]

# data
y1_early = []
y1_late = []
y2_early = []
y2_late = []
for part in hub.reordered_partitions:
    filtered_ids = [term_id for term_id in part if term_id in probe_term_ids_early + probe_term_ids_late]
    filtered_ids_early = [term_id for term_id in filtered_ids if term_id in probe_term_ids_early]
    filtered_ids_late = [term_id for term_id in filtered_ids if term_id in probe_term_ids_late]
    num_types_early = len(set(filtered_ids_early))
    num_types_late = len(set(filtered_ids_late))
    num_tokens_early = len(filtered_ids_early)
    num_tokens_late = len(filtered_ids_late)
    y1_early.append(num_types_early)
    y1_late.append(num_types_late)
    y2_early.append(num_tokens_early)
    y2_late.append(num_tokens_late)

# fig
fig, axarr = plt.subplots(dpi=DPI, nrows=2)
plt.suptitle('{}, block_order={}'.format(HUB_MODE, BLOCK_ORDER))
x = np.arange(NUM_PARTS)
y_names = ['types', 'tokens']
for ax, y_name, ys in zip(axarr, y_names, [(y1_early, y1_late), (y2_early, y2_late)]):
    ax.set_xlabel('Reordered Partition')
    ax.set_ylabel('Number of {}'.format(y_name))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    # ax.set_ylim([0, np.max(np.concatenate(ys))])
    ax.yaxis.grid(True, alpha=0.1)
    # plot
    ax.plot(x, np.mean(np.vstack(ys), axis=0), label='average', color='grey', linestyle=':')
    for y, label in zip(ys, ['{} ({})'.format(i, 'reordered' if IS_REORDERED else 'unordered')
                             for i in ['early', 'late']]):
        ax.plot(x, y, label=label)
# legend
for ax in axarr:
    ax.legend(frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
