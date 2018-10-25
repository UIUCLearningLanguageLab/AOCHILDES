import pandas as pd
import pyprind
import sys
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from childeshub.hub import Hub
# pip install git+https://github.com/phueb/CHILDESHub.git

HUB_MODE = 'syn'

CORPUS_NAME = 'childes-20180319'
MCDIP_PATH = 'mcdip.csv'
CONTEXT_SIZE = 7  # backwards only

hub = Hub(mode=HUB_MODE, corpus_name=CORPUS_NAME, part_order='inc_age', num_types=10000)

# t2mcdip (map target to its mcdip value)
df = pd.read_csv(MCDIP_PATH, index_col=False)
to_drop = []  # remove targets from df if not in vocab
for n, t in enumerate(df['target']):
    if t not in hub.train_terms.types:
        print('Dropping "{}"'.format(t))
        to_drop.append(n)
df = df.drop(to_drop)
targets = df['target'].values
mcdips = df['MCDIp'].values
t2mcdip = {t: mcdip for t, mcdip in zip(targets, mcdips)}

# collect context words of probes (if they are targets)
probe2context_tokens = {}
for p, cts in hub.probe_context_terms_dict.items():
    probe2context_tokens[p] = [ct for ct in cts if ct in targets]

# calculate result for each probe (average mcdip of context words weighted by number of times in target context)
res = {p: 0 for p in hub.probe_store.types}
for p, cts in probe2context_tokens.items():
    counter = Counter(cts)
    total_f = len(cts)
    res[p] = np.average([t2mcdip[ct] for ct in cts], weights=[counter[ct] / total_f for ct in cts])


def plot(xs, ys, xlabel, ylabel, annotations=None):
    fig, ax = plt.subplots(1, figsize=(7, 7), dpi=192)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    if annotations is not None:
        it = iter(annotations)
    for x, y in zip(xs, ys):
        ax.scatter(x, y, color='black')
        if annotations is not None:
            ax.annotate(next(it), (x + 0.005, y))
    # fit line
    plot_best_fit_line(ax, zip(xs, ys), 12)
    plt.tight_layout()
    plt.show()


def plot_best_fit_line(ax, xys, fontsize, color='red', zorder=3, x_pos=0.95, y_pos=0.1, plot_p=True):
    x, y = zip(*xys)
    try:
        best_fit_fxn = np.polyfit(x, y, 1, full=True)
    except Exception as e:  # cannot fit line
        print('rnnlab: Cannot fit line.', e)
        return
    slope = best_fit_fxn[0][0]
    intercept = best_fit_fxn[0][1]
    xl = [min(x), max(x)]
    yl = [slope * xx + intercept for xx in xl]
    # plot line
    ax.plot(xl, yl, linewidth=2, c=color, zorder=zorder)
    # plot rsqrd
    variance = np.var(y)
    residuals = np.var([(slope * xx + intercept - yy) for xx, yy in zip(x, y)])
    Rsqr = np.round(1 - residuals / variance, decimals=3)
    if Rsqr > 0.5:
        fontsize += 5
    ax.text(x_pos, y_pos, '$R^2$ = {}'.format(Rsqr), transform=ax.transAxes, fontsize=fontsize)
    if plot_p:
        p = np.round(linregress(x, y)[3], decimals=8)
        ax.text(x_pos, y_pos - 0.05, 'p = {}'.format(p), transform=ax.transAxes, fontsize=fontsize - 2)


probe_weighted_context_mcdip = [res[p] for p in hub.probe_store.types]
probe_median_cgs = [hub.calc_median_term_cg(p) for p in hub.probe_store.types]
probe_freqs = [hub.train_terms.term_freq_dict[p] for p in hub.probe_store.types]

plot(probe_median_cgs, np.log(probe_freqs),
     'probe_median_cgs', 'log probe_freqs')

plot(probe_weighted_context_mcdip, np.log(probe_freqs),
     'probe_weighted_context_mcdip', 'log probe_freqs', annotations=hub.probe_store.types)

plot(probe_weighted_context_mcdip, probe_median_cgs,
     'probe_weighted_context_mcdip', 'probe_median_cgs', annotations=hub.probe_store.types)
