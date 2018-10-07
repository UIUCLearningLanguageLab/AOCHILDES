import matplotlib.pyplot as plt
import seaborn as sns
import pyprind
import numpy as np

from hub import Hub

CORPUS_NAME = 'childes-20180319'
P_NOISE = 'early_2'  # TODO

MIN_CONTEXT_FREQ = 10
HUB_MODE = 'sem'
CONTEXT_DISTS = [1, 2, 3, 4, 5, 6]


def mkae_y_locs_dict(h, tokens_, context_dist):
    print('Calculating context stats with hub_mode={}...'.format(h.mode))
    cat_probes_expected_probs_d = {cat: np.array([1 / len(h.probe_store.cat_probe_list_dict[cat])
                                                  if probe in h.probe_store.cat_probe_list_dict[cat] else 0.0
                                                  for probe in h.probe_store.types])
                                   for cat in h.probe_store.cats}
    #
    #
    # cat_probes_expected_probs_d = {cat: np.array([h.train_terms.term_freq_dict[probe]
    #                                               if probe in h.probe_store.cat_probe_list_dict[cat] else 0.0
    #                                               for probe in h.probe_store.types])
    #                                for cat in h.probe_store.cats}
    # for k, v in cat_probes_expected_probs_d.items():  # normalize
    #     cat_probes_expected_probs_d[k] = v / v.sum()
    #     print(cat_probes_expected_probs_d[k].sum())

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
            # expected
            most_common_cat = sorted(h.probe_store.cats, key=lambda cat: context_d[context]['freq_by_cat'][cat])[-1]
            probes_expected_probs = cat_probes_expected_probs_d[most_common_cat]
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


# y_locs_dict
y_locs_dicts = []
hub = Hub(mode=HUB_MODE, block_order='inc_age', corpus_name=CORPUS_NAME, p_noise=P_NOISE)
for context_dist in CONTEXT_DISTS:
    y_locs_dict = mkae_y_locs_dict(hub, hub.reordered_tokens, context_dist)
    y_locs_dicts.append(y_locs_dict)


for n, (context_dist, y_locs_dict) in enumerate(zip(CONTEXT_DISTS, y_locs_dicts)):
    # fig
    _, ax = plt.subplots(dpi=192)
    ax.set_title('context-size={} P_NOISE={}, punct={}'.format(
        context_dist, P_NOISE, 'True' if CORPUS_NAME == 'childes-20180319' else 'False'))
    ax.set_ylabel('Probability')
    ax.set_xlabel('KL divergence')
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
    ax.hist(y1, density=True, label='partition 1', color=colors[0], histtype='step', bins=12, range=[0, 12])
    ax.hist(y2, density=True, label='partition 2', color=colors[1], histtype='step', bins=12, range=[0, 12])
    ax.text(0.02, 0.5, 'p1 mean={:.2f}+/-{:.1f}'.format(np.mean(y1), np.std(y1)), transform=ax.transAxes)
    ax.text(0.02, 0.4, 'p2 mean={:.2f}+/-{:.1f}'.format(np.mean(y2), np.std(y2)), transform=ax.transAxes)

    ax.axhline(y=0, color='grey')
    plt.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.show()

