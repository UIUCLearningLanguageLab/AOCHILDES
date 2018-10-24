import pandas as pd
import pyprind
import sys
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from childeshub.hub import Hub
# pip install git+https://github.com/phueb/CHILDESHub.git


CORPUS_NAME = 'childes-20180319'
MCDIP_PATH = 'mcdip.csv'
CONTEXT_SIZE = 7  # backwards only

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
        context = [ct for ct in hub.reordered_tokens[n - CONTEXT_SIZE: n] if ct in targets]
        target2context_tokens[t] += context

# calculate result for each target (average mcdip of context words weighted by number of times in target context)
res = {t: 0 for t in targets}
for t, cts in target2context_tokens.items():
    counter = Counter(cts)
    total_f = len(cts)
    res[t] = np.average([t2mcdip[ct] for ct in cts], weights=[counter[ct] / total_f for ct in cts])


s = sorted(res.items(), key=lambda i: i[1], reverse=True)
sorted_targets, sorted_res = zip(*s)

target_weighted_context_mcdip = [res[t] for t in targets]
target_median_cgs = [hub.calc_median_term_cg(t) for t in targets]

annotations = targets
fig, ax = plt.subplots(1, figsize=(7, 7), dpi=192)
ax.set_xlabel('target_weighted_context_mcdip', fontsize=12)
ax.set_ylabel('target_median_cgs', fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
if annotations is not None:
    it = iter(annotations)
for x, y in zip(target_weighted_context_mcdip, target_median_cgs):
    ax.scatter(x, y, color='black')
    if annotations is not None:
        ax.annotate(next(it), (x + 0.005, y))
plt.tight_layout()
plt.show()