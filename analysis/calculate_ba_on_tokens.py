import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import partial
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from childeshub.hub import Hub


"""
calculate ba on tokens - this allows comparison of amount of semantic category information present in partition 1 vs 2
"""

FIGSIZE = (8, 5)
YLIM = 0.65
TITLE_FONTSIZE = 10

NUM_SPLITS = 8

HUB_MODE = 'sem'
BPTT_STEPS = 1
DIRECTION = -1  # context is left if -1, context is right if +1


def calc_ba(probe_sims, probes, probe2cat, num_opt_init_steps=1, num_opt_steps=10):
    def calc_signals(_probe_sims, _labels, thr):  # vectorized algorithm is 20X faster
        probe_sims_clipped = np.clip(_probe_sims, 0, 1)
        probe_sims_clipped_triu = probe_sims_clipped[np.triu_indices(len(probe_sims_clipped), k=1)]
        predictions = np.zeros_like(probe_sims_clipped_triu, int)
        predictions[np.where(probe_sims_clipped_triu > thr)] = 1
        #
        tp = float(len(np.where((predictions == _labels) & (_labels == 1))[0]))
        tn = float(len(np.where((predictions == _labels) & (_labels == 0))[0]))
        fp = float(len(np.where((predictions != _labels) & (_labels == 0))[0]))
        fn = float(len(np.where((predictions != _labels) & (_labels == 1))[0]))
        return tp, tn, fp, fn

    # gold_mat
    if not len(probes) == probe_sims.shape[0] == probe_sims.shape[1]:
        raise RuntimeError(len(probes), probe_sims.shape[0], probe_sims.shape[1])
    num_rows = len(probes)
    num_cols = len(probes)
    gold_mat = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        probe1 = probes[i]
        for j in range(num_cols):
            probe2 = probes[j]
            if probe2cat[probe1] == probe2cat[probe2]:
                gold_mat[i, j] = 1

    # define calc_signals_partial
    labels = gold_mat[np.triu_indices(len(gold_mat), k=1)]
    calc_signals_partial = partial(calc_signals, probe_sims, labels)

    def calc_probes_ba(thr):
        tp, tn, fp, fn = calc_signals_partial(thr)
        specificity = np.divide(tn + 1e-7, (tn + fp + 1e-7))
        sensitivity = np.divide(tp + 1e-7, (tp + fn + 1e-7))  # aka recall
        ba = (sensitivity + specificity) / 2  # balanced accuracy
        return ba

    # use bayes optimization to find best_thr
    sims_mean = np.mean(probe_sims).item()
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}  # without this, warnings about predicted variance < 0
    bo = BayesianOptimization(calc_probes_ba, {'thr': (0.0, 1.0)}, verbose=False)
    bo.explore(
        {'thr': [sims_mean]})  # keep bayes-opt at version 0.6 because 1.0 occasionally returns 0.50 wrongly
    bo.maximize(init_points=num_opt_init_steps, n_iter=num_opt_steps,
                acq="poi", xi=0.01, **gp_params)  # smaller xi: exploitation
    best_thr = bo.res['max']['max_params']['thr']
    # use best_thr
    results = calc_probes_ba(best_thr)
    res = np.mean(results)
    return res


def plot_ba_trajs(part_id2y, part_id2x, title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Samples from Partition')
    ax.set_ylabel('Balanced Accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_ylim([0.5, YLIM])
    # plot
    for part_id, y in part_id2y.items():
        x = part_id2x[part_id]
        ax.plot(x, y, label='partition {}'.format(part_id + 1))
    #
    plt.legend(frameon=False, loc='upper left')
    plt.tight_layout()
    plt.show()


def calc_ba_from_windows(ws_mat, d):
    for window in ws_mat:
        first_word = hub.train_terms.types[window[0]]
        last_word = hub.train_terms.types[window[-1]]
        if DIRECTION == -1:  # context is defined to be words left of probe
            if last_word in hub.probe_store.types:
                for word_id in window[:-1]:
                    d[last_word][word_id] += 1
        elif DIRECTION == 1:
            if first_word in hub.probe_store.types:
                for word_id in window[0:]:
                    d[first_word][word_id] += 1
        else:
            raise AttributeError('Invalid arg to "DIRECTION".')
    # ba
    p_acts = np.asarray([d[p] for p in hub.probe_store.types])
    normalized_acts = normalize(p_acts, axis=1, norm='l1', copy=False)
    res = calc_ba(cosine_similarity(normalized_acts), hub.probe_store.types, probe2cat)
    return res


hub = Hub(mode=HUB_MODE, bptt_steps=BPTT_STEPS)
cats = hub.probe_store.cats
probe2cat = hub.probe_store.probe_cat_dict
vocab = hub.train_terms.types


#
part_ids = range(2)
part_id2bas = {part_id: [0.5] for part_id in part_ids}
part_id2num_windows = {part_id: [0] for part_id in part_ids}
for part_id in part_ids:
    # a window is [x1, x2, x3, x4, x5, x6, x7, y] if bptt=7
    windows_mat = hub.make_windows_mat(hub.reordered_partitions[part_id], hub.num_windows_in_part)
    print('shape of windows_mat={}'.format(windows_mat.shape))
    #
    xi = 0
    probe2act = {p: np.zeros(hub.params.num_types) for p in hub.probe_store.types}
    for windows_mat_chunk in np.vsplit(windows_mat, NUM_SPLITS):  # mimic incremental increase in ba
        ba = calc_ba_from_windows(windows_mat_chunk, probe2act)
        xi += len(windows_mat_chunk)
        part_id2bas[part_id].append(ba)
        part_id2num_windows[part_id].append(xi)
        print('part_id={} ba={:.3f}'.format(part_id, ba))
    print('------------------------------------------------------')


# plot
plot_ba_trajs(part_id2bas, part_id2num_windows,
              title='Semantic category information captured by CHILDES Bag-of-words model\n'
                    'context-size={} context-direction={}'.format(
                  BPTT_STEPS, 'left' if DIRECTION == -1 else 'right'))