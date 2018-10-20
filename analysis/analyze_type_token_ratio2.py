import matplotlib.pyplot as plt
import numpy as np
import pyprind

from childeshub.hub import Hub

HUB_MODE = 'syn'
BLOCK_ORDER = 'inc_age'
DPI = 192
NUM_PARTS = 2
IS_REORDERED = False  # true if lateness of probes is determined after partition reordering

CONTEXT_DIST = 3

hub = Hub(mode=HUB_MODE, num_parts=NUM_PARTS, part_order=BLOCK_ORDER)

# contexts
context_loc_d = {}
pbar = pyprind.ProgBar(hub.train_terms.num_tokens)
for loc, token in enumerate(hub.reordered_tokens[:-CONTEXT_DIST]):
    pbar.update()
    if token in hub.probe_store.types:
        context = tuple(hub.reordered_tokens[loc + d] for d in range(-CONTEXT_DIST, 0) if d != 0)
        try:
            context_loc_d[context].append(loc)
        except KeyError:
            context_loc_d[context] = []
            context_loc_d[context].append(loc)
sorted_contexts = sorted(context_loc_d.keys(), key=lambda k: np.mean(context_loc_d[k]))
num_contexts = len(sorted_contexts)
contexts_early = sorted_contexts[:num_contexts // 2]
contexts_late = sorted_contexts[-num_contexts // 2:]

print(contexts_early[:2])
print(len(contexts_early))
print(contexts_late[:2])
print(len(contexts_late))

# data
y1_early = []
y1_late = []
y2_early = []
y2_late = []
for part in hub.reordered_partitions:

    tokens = [hub.train_terms.types[term_id] for term_id in part]
    print('tokens calculated')
    contexts_in_part = [tuple(tokens[loc + d] for d in range(-CONTEXT_DIST, 0) if d != 0)
                        for loc, token in enumerate(tokens[:-CONTEXT_DIST])
                        if token in hub.probe_store.types]
    print('chunks calculated')

    contexts_in_part_early = [c for c in contexts_in_part if c in contexts_early]
    print('early done')
    contexts_in_part_late = [c for c in contexts_in_part if c in contexts_late]
    print('late done')
    num_types_early = len(set(contexts_in_part_early))
    num_types_late = len(set(contexts_in_part_late))
    num_tokens_early = len(contexts_in_part_early)
    num_tokens_late = len(contexts_in_part_late)
    y1_early.append(num_types_early)
    y1_late.append(num_types_late)
    y2_early.append(num_tokens_early)
    y2_late.append(num_tokens_late)

# fig
fig, axarr = plt.subplots(dpi=DPI, nrows=2)
plt.suptitle(HUB_MODE)
x = np.arange(NUM_PARTS)
y_names = ['Type', 'Token']
for ax, y_name, ys in zip(axarr, y_names, [(y1_early, y1_late), (y2_early, y2_late)]):
    ax.set_xlabel('Partition')
    ax.set_ylabel('{} Frequency'.format(y_name))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    # ax.set_ylim([0, np.max(np.concatenate(ys))])
    ax.yaxis.grid(True, alpha=0.1)
    # plot
    ax.plot(x, np.mean(np.vstack(ys), axis=0), label='average', color='grey', linestyle=':')
    for y, label in zip(ys, ['{} {}'.format(i, '(reordered)' if IS_REORDERED else '')
                             for i in ['early contexts', 'late contexts']]):
        ax.plot(x, y, label=label + ' (size={})'.format(CONTEXT_DIST))
# legend
for ax in axarr:
    ax.legend(frameon=False)
fig.tight_layout()
plt.show()
