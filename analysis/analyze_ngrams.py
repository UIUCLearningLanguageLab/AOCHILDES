
from childeshub.hub import Hub


NGRAM_SIZE = 2
HUB_MODE = 'sem'

print('N-gram Size: {}'.format(NGRAM_SIZE))
print()

hub = Hub(mode=HUB_MODE)

tokens_early = hub.first_half_tokens
tokens_late = hub.second_half_tokens
num_tokens_early = len(tokens_early)
num_tokens_late = len(tokens_late)
print('Num tokens')
print('{:,} {:,}'.format(num_tokens_early, num_tokens_late))
print()

probes_in_tokens_early = [token for token in tokens_early
                          if token in hub.probe_store.types]
probes_in_tokens_late = [token for token in tokens_late
                         if token in hub.probe_store.types]
num_total_probes_early = len(probes_in_tokens_early)
num_total_probes_late = len(probes_in_tokens_late)
print('Num total probes')
print('{:,} {:,}'.format(num_total_probes_early, num_total_probes_late))
print()


ngrams_early = hub.get_sliding_windows(NGRAM_SIZE, tokens_early)
ngrams_late = hub.get_sliding_windows(NGRAM_SIZE, tokens_late)
num_ngrams_early = len(ngrams_early)
num_ngrams_late = len(ngrams_late)
print('Num total ngrams')
print('{:,} {:,}'.format(num_ngrams_early, num_ngrams_late))
print()

unique_ngrams_early = set(ngrams_early)
ngram_set_len_early = len(unique_ngrams_early)
unique_ngrams_late = set(ngrams_late)
ngram_set_len_late = len(unique_ngrams_late)
print('Num unique n-grams')
print('{:,} {:,}'.format(ngram_set_len_early, ngram_set_len_late))
print()

print('num n-grams in 1 also in 2:')
print(len([ngram for ngram in unique_ngrams_early if ngram in unique_ngrams_late]))
print()

unique_ngrams_late.update(unique_ngrams_early)
num_updated_ngrams = len(unique_ngrams_late)
print('Num combined unique ngrams')
print(num_updated_ngrams)
print('Percent of combined unique {}-grams'.format(NGRAM_SIZE))
print(ngram_set_len_early / num_updated_ngrams, ngram_set_len_late / num_updated_ngrams)
print()

unique_probe_ngrams_early = set([ngram for ngram in ngrams_early if any([t in hub.probe_store.types for t in ngram])])
ngram_probes_set_len_early = len(unique_probe_ngrams_early)
uniq_probe_ngrams_late = set([ngram for ngram in ngrams_late if any([t in hub.probe_store.types for t in ngram])])
ngram_probes_set_len_late = len(uniq_probe_ngrams_late)
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

num_ngrams_by_probe_count_early = num_ngrams_early / num_total_probes_early
num_ngrams_by_probe_count_late = num_ngrams_late / num_total_probes_late
print('Num total ngrams containing probes / num total probes in tokens')
print('{:.2f} {:.2f}'.format(num_ngrams_by_probe_count_early, num_ngrams_by_probe_count_late))
print()

