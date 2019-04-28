import numpy as np
import matplotlib.pyplot as plt

from childeshub.hub import Hub

FIGSIZE = (6, 6)
TITLE_FONTSIZE = 10

HUB_MODE = 'sem'


hub = Hub(mode=HUB_MODE)
cats = hub.probe_store.cats
probe2cat = hub.probe_store.probe_cat_dict
vocab = hub.train_terms.types


def plot_tmp(xs, ys, title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=None)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Repetition')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_ylim([0, 20])  # rep
    ax.set_xlim([0, 4])  # cov
    # plot
    for x, y in zip(xs, ys):
        ax.plot([0, x], [y, 0])
    #
    plt.tight_layout()
    plt.show()


for half_id, tokens in enumerate([hub.first_half_tokens, hub.second_half_tokens]):
    # init
    print('Counting...')
    cats = set(probe2cat.values())
    word2prev_cat2count = {w: {cat: 0 for cat in cats} for w in vocab}
    word2prev_word2count = {w: {w: 0 for w in vocab} for w in vocab}
    for n, token in enumerate(tokens[1:]):
        prev_token = tokens[n - 1]
        try:
            prev_cat = probe2cat[prev_token]
        except KeyError:
            continue
        #
        word2prev_cat2count[token][prev_cat] += 1
        word2prev_word2count[token][prev_token] += 1

    coverages = []
    repetitions = []
    for cat in cats:
        cat_probes = [p for p in hub.probe_store.types if probe2cat[p] == cat]
        cat_repetitions = []
        cat_coverages = []
        for word in vocab:
            repetition = word2prev_cat2count[word][cat]  # how many times word occurs after category member
            coverage = np.count_nonzero([word2prev_word2count[word][w] for w in cat_probes])
            if repetition > 0.0:
                cat_repetitions.append(repetition)
                cat_coverages.append(coverage)

        rep = np.mean(cat_repetitions)
        cov = np.mean(cat_coverages)
        print(rep, cov)
        repetitions.append(rep)
        coverages.append(cov)

    plot_tmp(coverages, repetitions,
             title='partition {}'.format(half_id + 1))
    print('------------------------------------------------------')