import matplotlib.pyplot as plt
import seaborn as sns
import pyprind
import numpy as np

from childeshub.hub import Hub

CORPUS_NAME = 'childes-20180315'

MIN_CONTEXT_FREQ = 10
HUB_MODE = 'sem'
CONTEXT_DISTS = [1, 2, 3, 4, 5, 6]


def make_y_locs_dict(h, tokens_, context_dist):
    # context_d
    context_d = {}
    pbar = pyprind.ProgBar(h.train_terms.num_tokens)
    for loc, token in enumerate(tokens_[:-context_dist]):
        pbar.update()
        context = tuple(tokens_[loc + d] for d in range(-context_dist, 0) if d != 0)
        if token in h.probe_store.types:
            try:
                cat = h.probe_store.probe_cat_dict[token]
                context_d[context]['freq_by_probe'][token] += 1
                context_d[context]['freq_by_cat'][cat] += 1
                context_d[context]['total_freq'] += 1
                context_d[context]['term_freq'] += 0
                context_d[context]['probe_freq'] += 1
                context_d[context]['locs'].append(loc)
            except KeyError:
                context_d[context] = {'freq_by_probe': {probe: 0.0 for probe in h.probe_store.types},
                                      'freq_by_cat': {cat: 0.0 for cat in h.probe_store.cats},
                                      'total_freq': 0,
                                      'term_freq': 0,
                                      'probe_freq': 0,
                                      'locs': [],
                                      'y': 0}  # must be numeric for sorting
        else:
            try:
                context_d[context]['term_freq'] += 1
            except KeyError:  # only update contexts which are already tracked
                pass

    # result
    result = {}
    for context in context_d.keys():
        context_freq = context_d[context]['total_freq']
        if context_freq > MIN_CONTEXT_FREQ:
            num_most_common_cat_observed = sorted(context_d[context]['freq_by_cat'].items(),
                                                  key=lambda i: i[1])[-1][1]
            y = num_most_common_cat_observed / (context_d[context]['probe_freq'] + context_d[context]['term_freq'])
            # collect y
            result[y] = context_d[context]['locs']
    return result


y_locs_dicts = []
hub = Hub(mode=HUB_MODE, part_order='inc_age', corpus_name=CORPUS_NAME)
for context_dist in CONTEXT_DISTS:
    y_locs_dict = make_y_locs_dict(hub, hub.reordered_tokens, context_dist)
    y_locs_dicts.append(y_locs_dict)


for n, (context_dist, y_locs_dict) in enumerate(zip(CONTEXT_DISTS, y_locs_dicts)):
    # fig
    _, ax = plt.subplots(dpi=192)
    ax.set_title('context-size={}, punct={}'.format(
        context_dist, 'True' if CORPUS_NAME == 'childes-20180319' else 'False'))
    ax.set_ylabel('Probability')
    ax.set_xlabel('Category Restrictiveness')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    # plot
    colors = sns.color_palette("hls", 2)[::-1]
    y1 = []
    y2 = []
    for y, locs in y_locs_dict.items():
        num_locs_in_part1 = len(np.where(np.array(locs) < hub.midpoint_loc)[0])
        num_locs_in_part2 = len(np.where(np.array(locs) > hub.midpoint_loc)[0])
        y1 += [y] * num_locs_in_part1
        y2 += [y] * num_locs_in_part2
    y1 = np.array(y1)
    y2 = np.array(y2)
    ax.hist(y1, density=True, label='partition 1', color=colors[0], histtype='step', bins=100, range=[0, 1])
    ax.hist(y2, density=True, label='partition 2', color=colors[1], histtype='step', bins=100, range=[0, 1])
    ax.text(0.7, 0.5, 'p1 mean={:.2f}+/-{:.1f}'.format(np.mean(y1), np.std(y1)), transform=ax.transAxes)
    ax.text(0.7, 0.4, 'p2 mean={:.2f}+/-{:.1f}'.format(np.mean(y2), np.std(y2)), transform=ax.transAxes)
    ax.axhline(y=0, color='grey')
    plt.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.show()

