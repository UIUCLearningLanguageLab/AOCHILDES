import numpy as np

from childeshub.hub import Hub

HUB_MODE = 'sem'


hub = Hub(mode=HUB_MODE)
cats = hub.probe_store.cats
probe2cat = hub.probe_store.probe_cat_dict
vocab = hub.train_terms.types


for tokens in [hub.first_half_tokens, hub.second_half_tokens]:
    # init
    print('Counting...')
    word2cat2count = {w: {cat: 0 for cat in cats} for w in vocab}
    word2count = {t: 0 for t in vocab}
    for n, token in enumerate(tokens[1:]):
        prev_token = tokens[n-1]
        try:
            prev_cat = probe2cat[prev_token]
        except KeyError:
            prev_cat = None
        #
        word2cat2count[token][prev_cat] += 1
        word2count[token] += 1

    print('Calculating coverage ratios...')
    cat_mean_ratios = []
    cat_var_ratios = []
    for cat in cats:
        num_after_cats = []
        num_totals = []
        for word in vocab:
            num_after_cat = word2cat2count[word][cat]
            num_total = word2count[word] + 1
            ratio = num_after_cat / num_total
            if num_after_cat > 0:  # only interested in words that influence category
                num_after_cats.append(num_after_cat)
                num_totals.append(num_total)
        #
        ratios = [a / b for a, b in zip(num_after_cats, num_totals)]
        cat_mean_ratio = np.mean(ratios)
        cat_var_ratio = np.var(ratios)
        #
        cat_mean_ratios.append(cat_mean_ratio)
        cat_var_ratios.append(cat_var_ratio)
        print(cat, cat_mean_ratio.round(3), cat_var_ratio.round(5), len(num_totals), 'max={}'.format(np.max(num_after_cats)))

    print(np.mean(cat_mean_ratios), np.mean(cat_var_ratios))
    print(1 / hub.probe_store.num_cats)
    print('-----------------------------------------------')