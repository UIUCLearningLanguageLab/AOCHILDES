from collections import Counter
from itertools import chain
import numpy as np

from childeshub.hub import Hub

DPI = 192
HUB_MODE = 'sem'

MAX_NUM_CONTEXTS = 10
CONTEXT_SIZE = 3


hub = Hub(mode=HUB_MODE)


def make_contexts_dicts(tokens):  # left context
    cat2contexts = {cat: [] for cat in hub.probe_store.cats}
    probe2contexts = {probe: [] for probe in hub.probe_store.types}
    for n, t in enumerate(tokens):
        if t in hub.probe_store.types:
            context = tuple(tokens[n - CONTEXT_SIZE:n])
            cat = hub.probe_store.probe_cat_dict[t]
            cat2contexts[cat].append(context)
            probe2contexts[t].append(context)
    return cat2contexts, probe2contexts


# get most category-diagnostic category contexts (occurring more often with category than not)
cat2diagnostic_contexts = {}
cat2contexts, probe2contexts = make_contexts_dicts(hub.reordered_tokens)
total_counter = Counter(list(chain(*[contexts for contexts in cat2contexts.values()])))
for cat, cs in cat2contexts.items():
    cat_counter = Counter(cs)
    diag_contexts = [c for c in set(cs) if cat_counter[c] > total_counter[c] - cat_counter[c]]
    cat2diagnostic_contexts[cat] = diag_contexts

# for each probe, get number of contexts and number of category-diagnostic contexts
probe2diag_per_all = {}
probe2diag = {}
for probe in hub.probe_store.types:
    cat = hub.probe_store.probe_cat_dict[probe]
    all_probe_contexts = probe2contexts[probe]
    diag_cat_contexts = cat2diagnostic_contexts[cat]
    #
    num_cat_diag_probe_contexts = np.sum([c in diag_cat_contexts for c in probe2contexts[probe]])
    num_all_probe_contexts = len(all_probe_contexts)
    #
    probe2diag[probe] = num_cat_diag_probe_contexts
    probe2diag_per_all[probe] = num_cat_diag_probe_contexts / num_all_probe_contexts
    #
    print('{:<12} {:>12,} {:>12,}'.format(probe, num_cat_diag_probe_contexts, num_all_probe_contexts))

print('Sorted by ratio of number of category-diagnostic contexts to all contexts')
s = sorted(hub.probe_store.types, key=probe2diag_per_all.get)
for probe in s:
    print(probe, probe2diag_per_all[probe])

print('Sorted by absolute number of category-diagnostic contexts')
s = sorted(hub.probe_store.types, key=probe2diag.get)
for probe in s:
    print(probe, probe2diag[probe])