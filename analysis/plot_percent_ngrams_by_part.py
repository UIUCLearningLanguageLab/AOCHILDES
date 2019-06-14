import matplotlib.pyplot as plt

from childeshub.hub import Hub

# fig
PLOT_NUM_SVS = 64
FONTSIZE = 18
FIGSIZE = (8, 8)
DPI = None

MAX_NGRAM_SIZE = 7
HUB_MODE = 'sem'


def plot_comparison(d):
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('', fontsize=FONTSIZE)
    ax.set_ylabel('Percent of unique n-grams in corpus', fontsize=FONTSIZE)
    ax.set_xlabel('n-gram size', fontsize=FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    for part_id, y in d.items():
        x = range(1, len(y) + 1)
        ax.plot(x, y, label='partition {}'.format(part_id), linewidth=2)
    ax.legend(loc='best', frameon=False, fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()


hub = Hub(mode=HUB_MODE)

tokens_early = hub.first_half_tokens
tokens_late = hub.second_half_tokens

ngram_sizes = list(range(1, MAX_NGRAM_SIZE + 1))
part_id2y = {1: [], 2: []}
for ngram_size in ngram_sizes:
    # make n-grams
    ngrams_early = hub.get_sliding_windows(ngram_size, tokens_early)
    ngrams_late = hub.get_sliding_windows(ngram_size, tokens_late)
    num_ngrams_early = len(ngrams_early)
    num_ngrams_late = len(ngrams_late)

    # get unique n-grams
    unique_ngrams1 = set(ngrams_early)
    ngram_set_len1 = len(unique_ngrams1)
    unique_ngrams2 = set(ngrams_late)
    ngram_set_len2 = len(unique_ngrams2)

    unique_ngrams2.update(unique_ngrams1)
    num_updated_ngrams = len(unique_ngrams2)
    print('Num combined unique ngrams')
    print(num_updated_ngrams)
    print('Percent of combined unique {}-grams'.format(ngram_size))
    yi1 = ngram_set_len1 / num_updated_ngrams
    yi2 = ngram_set_len2 / num_updated_ngrams
    print(yi1, yi2)
    print()

    # collect
    part_id2y[1].append(yi1)
    part_id2y[2].append(yi2)

plot_comparison(part_id2y)  # sharing of n-grams between partitions is proportional to the percentage shown in plot
