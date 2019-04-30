import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import partial
from bayes_opt import BayesianOptimization

from childeshub.hub import Hub


"""
calculate ba on tokens - this allows comparison of amount of semantic category information present in partition 1 vs 2
"""


HUB_MODE = 'sem'
BPTT_STEPS = 7


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


hub = Hub(mode=HUB_MODE, bptt_steps=BPTT_STEPS)
cats = hub.probe_store.cats
probe2cat = hub.probe_store.probe_cat_dict
vocab = hub.train_terms.types


for part_id in range(2):
    # a window is [x1, x2, x3, x4, x5, x6, x7, y] if bptt=7
    windows_mat = hub.make_windows_mat(hub.reordered_partitions[part_id], hub.num_windows_in_part)
    print('shape of windows_mat={}'.format(windows_mat.shape))

    # probe2act
    probe2act = {p: np.zeros(hub.params.num_types) for p in hub.probe_store.types}
    for window in windows_mat:
        first_word = hub.train_terms.types[window[-1]]
        if first_word in hub.probe_store.types:
            for word_id in window[:-1]:
                probe2act[first_word][word_id] += 1
    # ba
    p_acts = [probe2act[p] for p in hub.probe_store.types]
    ba = calc_ba(cosine_similarity(p_acts), hub.probe_store.types, probe2cat)
    print('part_id={} ba={:.3f}'.format(part_id, ba))
    print('------------------------------------------------------')