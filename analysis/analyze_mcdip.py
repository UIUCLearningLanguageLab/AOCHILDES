import pandas as pd
import pyprind
import sys
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from childeshub.hub import Hub
# pip install git+https://github.com/phueb/CHILDESHub.git


CORPUS_NAME = 'childes-20180319'
MCDIP_PATH = 'mcdip.csv'
CONTEXT_SIZE = 16  # bidirectional

hub = Hub(corpus_name=CORPUS_NAME, part_order='inc_age', num_types=10000)

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

# collect context words of targets
print('Collecting context words...')
target2context_tokens = {t: [] for t in targets}
pbar = pyprind.ProgBar(hub.train_terms.num_tokens, stream=sys.stdout)
for n, t in enumerate(hub.reordered_tokens):
    pbar.update()
    if t in targets:
        context_left = [ct for ct in hub.reordered_tokens[n - CONTEXT_SIZE: n] if ct in targets]
        context_right = [ct for ct in hub.reordered_tokens[n + 1: n + 1 + CONTEXT_SIZE] if ct in targets]
        target2context_tokens[t] += context_left + context_right

# calculate result for each target (average mcdip of context words weighted by number of times in target context)
res = {t: 0 for t in targets}
for t, cts in target2context_tokens.items():
    counter = Counter(cts)
    total_f = len(cts)
    res[t] = np.average([t2mcdip[ct] for ct in cts], weights=[counter[ct] / total_f for ct in cts])


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


def plot_best_fit_line(ax, xys, fontsize, color='red', zorder=3, x_pos=0.85, y_pos=0.1, plot_p=True):
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


target_weighted_context_mcdip = [res[t] for t in targets]
target_median_cgs = [hub.calc_median_term_cg(t) for t in targets]
target_mcdips = [t2mcdip[t] for t in targets]
target_freqs = [hub.train_terms.term_freq_dict[t] for t in targets]

plot(target_weighted_context_mcdip, target_mcdips,
     'KWOOC', 'MCDIp')

plot(target_median_cgs, np.log(target_freqs),
     'target_median_cgs', 'target_freqs')

plot(target_weighted_context_mcdip, np.log(target_freqs),
     'target_weighted_context_mcdip', 'target_freqs')

plot(target_mcdips, np.log(target_freqs),
     'target_mcdips', ' log target_freqs')

plot(target_weighted_context_mcdip, target_median_cgs,
     'target_weighted_context_mcdip', 'target_median_cgs')
