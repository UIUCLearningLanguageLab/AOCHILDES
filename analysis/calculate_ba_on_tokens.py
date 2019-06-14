import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from cytoolz import itertoolz
from functools import partial
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

from childeshub.hub import Hub


"""
calculate ba on tokens.
this allows comparison of amount of semantic category information present in partition 1 vs 2
"""

FIGSIZE = (8, 5)
YLIM = 0.65
TITLE_FONTSIZE = 10

HUB_MODE = 'sem'
NUM_BA_EVALS = 2
WINDOW_SIZE = 2


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
    ax.set_xlabel('Number of Words in Partition')
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


hub = Hub(mode=HUB_MODE)
probe2cat = hub.probe_store.probe_cat_dict

part_ids = range(2)
part_id2bas = {part_id: [0.5] for part_id in part_ids}
part_id2num_windows = {part_id: [0] for part_id in part_ids}
for part_id in part_ids:
    tokens = [hub.first_half_tokens, hub.second_half_tokens][part_id]
    xi = 0
    num_tokens_in_chunk = len(tokens) // NUM_BA_EVALS
    for tokens_chunk in itertoolz.partition_all(num_tokens_in_chunk, tokens):  # mimic incremental increase in ba
        tw_mat, xws, yws = hub.make_term_by_window_co_occurrence_mat(
            tokens=tokens_chunk, window_size=WINDOW_SIZE, probes_in_windows_only=False)  # TODO

        # TODO to match analysis in tree-transitions, probes should be in x-words - doesn't work

        # there are multiple windows with probe in last position when window_size > 1
        # probe_reps = []
        # for probe in hub.probe_store.types:
        #     col_bools = [True if xw == probe else False for xw in xws]
        #     probe_rep = np.asarray(np.mean(tw_mat[:, col_bools], axis=1)).flatten()
        #     # print(probe, np.count_nonzero(col_bools), probe_rep.shape)
        #     probe_reps.append(probe_rep)

        # TODO old way - probes are in yws - this works
        filtered_probes = [probe for probe in hub.probe_store.types if probe in yws]
        row_ids = [yws.index(probe) for probe in filtered_probes]
        probe_reps = tw_mat[row_ids, :].toarray()


        # ba
        print('shape of probe_reps={}'.format(probe_reps.shape))
        ba = calc_ba(cosine_similarity(probe_reps), filtered_probes, probe2cat)
        # collect
        xi += len(tokens_chunk)
        part_id2bas[part_id].append(ba)
        part_id2num_windows[part_id].append(xi)
        print('part_id={} ba={:.3f}'.format(part_id, ba))
    print('------------------------------------------------------')


# plot
plot_ba_trajs(part_id2bas, part_id2num_windows,
              title='Semantic category information in AO-CHILDES'
                    '\ncaptured by term-window co-occurrence matrix\n'
                    'with window-size={}'.format(WINDOW_SIZE))