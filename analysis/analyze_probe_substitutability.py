import matplotlib.pyplot as plt
import seaborn as sns
import pyprind
import numpy as np
from scipy.stats import ttest_ind

from childeshub.hub import Hub

CORPUS_NAME = 'childes-20180319'

MIN_CONTEXT_FREQ = 10
MIN_CAT_FREQ = 1
HUB_MODE = 'sem'
CONTEXT_DISTS = [1]
YMAX = 0.5


def make_y2locs(h, tokens_, context_dist):
    print('Calculating context stats with hub_mode={}...'.format(h.mode))
    cat2probes_expected_probs = {cat: np.array([1 / len(h.probe_store.cat_probe_list_dict[cat])
                                                if probe in h.probe_store.cat_probe_list_dict[cat] else 0.0
                                                for probe in h.probe_store.types])
                                 for cat in h.probe_store.cats}
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
            # observed
            probes_observed_probs = np.array(
                [context_d[context]['freq_by_probe'][probe] / context_d[context]['total_freq']
                 for probe in h.probe_store.types])
            # compute KL div for each category that the context is associated with (not just most common category)
            for cat, cat_freq in context_d[context]['freq_by_cat'].items():
                if cat_freq > MIN_CAT_FREQ:
                    probes_expected_probs = cat2probes_expected_probs[cat]
                    y = calc_kl_divergence(probes_expected_probs, probes_observed_probs)  # asymmetric
                    # collect y
                    result[y] = context_d[context]['locs']
    return result


def calc_kl_divergence(p, q, epsilon=0.00001):
    pe = p + epsilon
    qe = q + epsilon
    divergence = np.sum(pe * np.log2(pe / qe))
    return divergence


def print_contexts(d):
    for context, infos in sorted(d.items(),
                                 key=lambda i: (i[1]['y'], i[1]['probe_freq'])):  # sort by y, then by context_freq
        if d[context]['probe_freq'] > MIN_CONTEXT_FREQ:
            print(round(infos['y'], 2),
                  '{}'.format(context),
                  d[context]['probe_freq'],
                  sorted(d[context]['freq_by_cat'].items(), key=lambda i: i[1])[-3:],
                  sorted(d[context]['freq_by_probe'].items(), key=lambda i: i[1])[-5:])


# data
y2locs_list = []
hub = Hub(mode=HUB_MODE, part_order='inc_age', corpus_name=CORPUS_NAME)
for context_dist in CONTEXT_DISTS:
    y2locs = make_y2locs(hub, hub.reordered_tokens, context_dist)
    y2locs_list.append(y2locs)

# plot
for context_dist, y2locs in zip(CONTEXT_DISTS, y2locs_list):
    # fig
    fontsize = 16
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('context-size={}'.format(context_dist), fontsize=fontsize)
    ax.set_ylabel('Probability', fontsize=fontsize)
    ax.set_xlabel('KL Divergence', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.set_ylim([0, YMAX])
    # plot
    colors = sns.color_palette("hls", 2)[::-1]
    y1 = []
    y2 = []
    for y, locs in y2locs.items():
        num_locs_in_part1 = len(np.where(np.array(locs) < hub.midpoint_loc)[0])
        num_locs_in_part2 = len(np.where(np.array(locs) > hub.midpoint_loc)[0])
        y1 += [y] * num_locs_in_part1
        y2 += [y] * num_locs_in_part2
    y1 = np.array(y1)
    y2 = np.array(y2)
    num_bins = 20
    y1binned, x1, _ = ax.hist(y1, density=True, label='partition 1', color=colors[0], histtype='step',
                              bins=num_bins, range=[0, 12], zorder=3)
    y2binned, x2, _ = ax.hist(y2, density=True, label='partition 2', color=colors[1], histtype='step',
                              bins=num_bins, range=[0, 12], zorder=3)
    ax.text(0.02, 0.7, 'partition 1:\nmean={:.2f}+/-{:.1f}\nn={:,}'.format(
        np.mean(y1), np.std(y1), len(y1)), transform=ax.transAxes, fontsize=fontsize - 2)
    ax.text(0.02, 0.55, 'partition 2:\nmean={:.2f}+/-{:.1f}\nn={:,}'.format(
        np.mean(y2), np.std(y2), len(y2)), transform=ax.transAxes, fontsize=fontsize - 2)
    #  fill between the lines (highlighting the difference between the two histograms)
    for i, x1i in enumerate(x1[:-1]):
        y1line = [y1binned[i], y1binned[i]]
        y2line = [y2binned[i], y2binned[i]]
        ax.fill_between(x=[x1i, x1[i + 1]],
                        y1=y1line,
                        y2=y2line,
                        where=y1line > y2line,
                        color=colors[0],
                        alpha=0.5,
                        zorder=2)
    #
    plt.legend(frameon=False, loc='upper left', fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    # t test
    t, prob = ttest_ind(y1, y2, equal_var=False)
    print('t={}'.format(t))
    print('p={:.6f}'.format(prob))
    print()

