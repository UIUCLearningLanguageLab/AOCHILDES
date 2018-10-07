
from hub import Hub

CORPUS_NAME = 'childes-20171212'
PROBES_NAME = 'semantic-raw'

NUM_SPLITS = 8
NGRAM_RANGE = (2, 3)
HUB_MODE = 'sem'

print('N-gram Range: {}'.format(NGRAM_RANGE))
print()

hub = Hub(mode=HUB_MODE, corpus_name=CORPUS_NAME, probes_name=PROBES_NAME)

idx = hub.train_terms.num_tokens // NUM_SPLITS
tokens_early = hub.reordered_token_ids[:idx]
tokens_late = hub.reordered_token_ids[-idx:]
num_tokens_early = len(tokens_early)
num_tokens_late = len(tokens_late)
print('Num tokens')
print('{:,} {:,}'.format(num_tokens_early, num_tokens_late))
print()

probes_in_tokens_early = [token for token in tokens_early
                          if token in hub.probe_store.types]
probes_in_tokens_late = [token for token in tokens_late
                         if token in hub.probe_store.types]
num_total_probes_early_early = len(probes_in_tokens_early)
num_total_probes_late = len(probes_in_tokens_late)
print('Num total probes')
print('{:,} {:,}'.format(num_total_probes_early_early, num_total_probes_late))
print()


ngrams_early = hub.get_ngrams(NGRAM_RANGE, tokens_early)
ngrams_late = hub.get_ngrams(NGRAM_RANGE, tokens_late)
num_ngrams_early = len(ngrams_early)
num_ngrams_late = len(ngrams_late)
print('Num total ngrams')
print('{:,} {:,}'.format(num_ngrams_early, num_ngrams_late))
print()

ngram_set_len_early = len(set(ngrams_early))
ngram_set_len_late = len(set(ngrams_late))
print('Num unique n-grams')
print('{:,} {:,}'.format(ngram_set_len_early, ngram_set_len_late))
print()

ngram_probes_set_len_early = len(
    set([ngram for ngram in ngrams_early
         if any([t in hub.probe_store.types for t in ngram.split()])]))
ngram_probes_set_len_late = len(
    set([ngram for ngram in ngrams_late
         if any([t in hub.probe_store.types for t in ngram.split()])]))
print('Num unique n-grams containing probes')
print('{:,} {:,}'.format(ngram_probes_set_len_early, ngram_probes_set_len_late))
print()


probes_set_len_early = len(set(probes_in_tokens_early))
probes_set_len_late = len(set(probes_in_tokens_late))
ngram_set_len_by_probe_set_len_early = ngram_probes_set_len_early / probes_set_len_late
ngram_set_len_by_probe_set_len_late = ngram_probes_set_len_late / probes_set_len_early
print('Num unique ngrams containing probes / num unique probes in tokens')
print('{:.2f} {:.2f}'.format(ngram_set_len_by_probe_set_len_early, ngram_set_len_by_probe_set_len_late))
print()

num_ngrams_by_probe_count_early = num_ngrams_early / num_total_probes_early_early
num_ngrams_by_probe_count_late = num_ngrams_late / num_total_probes_late
print('Num total ngrams containing probes / num total probes in tokens')
print('{:.2f} {:.2f}'.format(num_ngrams_by_probe_count_early, num_ngrams_by_probe_count_late))
print()