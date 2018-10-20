import numpy as np

from childeshub.hub import Hub

LATENESS_THR = 0.95
VERBOSE = False

CORPUS_NAME = 'childes-20171212'
PROBES_NAME = 'semantic-complete'
HUB_MODE = 'sem'
NUM_REFS = 100
NUM_MOST_EXTREME = 50
TO_ANALYZE = 'terms'
NUM_ANALYZE = 500


hub = Hub(mode=HUB_MODE, corpus_name=CORPUS_NAME, probes_name=PROBES_NAME)

if TO_ANALYZE == 'probes':
    to_analyze = hub.probe_store.types
elif TO_ANALYZE == 'terms':
    to_analyze = list(zip(*hub.train_terms.term_freq_dict.most_common(NUM_ANALYZE)))[0]
else:
    raise AttributeError('Invalid arg to "TO_ANALYZE".')

refs = np.random.choice(to_analyze, NUM_REFS, replace=False).tolist()

if TO_ANALYZE == 'probes':
    cat_lateness_dict = {}
    for cat in hub.probe_store.cats:
        print(cat)
        cat_lateness_sum = 0
        num_cat_probes = len(hub.probe_store.cat_probe_list_dict[cat])
        for term in hub.probe_store.cat_probe_list_dict[cat]:
            cat_lateness_sum += hub.calc_lateness(term, is_probe=False)
        cat_lateness = round(cat_lateness_sum / num_cat_probes, 2)
        cat_lateness_dict[cat] = cat_lateness
        print(cat_lateness)
    print()
    for k, v in sorted(cat_lateness_dict.items(), key=lambda i: i[1]):
        print(k, v)

sorted_by_avg_loc = sorted(to_analyze, key=hub.calc_avg_reordered_loc)

print('Earliest:')
num_chars = 0
for term in sorted_by_avg_loc[:NUM_MOST_EXTREME]:
    print(term)
    num_chars += len(term)
print(num_chars / NUM_MOST_EXTREME)

print('Latest:')
num_chars = 0
for term in sorted_by_avg_loc[-NUM_MOST_EXTREME:]:
    print(term)
    num_chars += len(term)
print(num_chars / NUM_MOST_EXTREME)
