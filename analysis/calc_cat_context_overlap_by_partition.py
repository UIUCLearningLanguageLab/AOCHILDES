import multiprocessing as mp

from childeshub.hub import Hub

HUB_MODE = 'sem'
MAX_DISTANCE = 3   # TODO


hub = Hub(mode=HUB_MODE)


def calc_overlap(d, cat):
    """
    quantify overlap between contexts of one category with contexts of all other categories
    """
    other_cats = [c for c in hub.probe_store.cats if c != cat]
    num = 0
    num_total = 0
    for context_word1 in d[cat]:
        for other_cat in other_cats:
            for context_word2 in d[other_cat]:
                if context_word1 == context_word2:
                    num += 1
                num_total += 1

    res = num / num_total  # division controls for greater number of probes in partition 1
    print(res)
    return res


for part_id, tokens in enumerate([hub.first_half_tokens, hub.second_half_tokens]):
    #
    cat2contexts = {cat: [] for cat in hub.probe_store.cats}
    for loc, token in enumerate(tokens):
        if token in hub.probe_store.types:
            cat = hub.probe_store.probe_cat_dict[token]
            for dist in range(1, MAX_DISTANCE + 1):
                cat2contexts[cat].append(tokens[loc + dist])
    #
    pool = mp.Pool(processes=4)
    results = [pool.apply_async(calc_overlap, args=(cat2contexts, cat)) for cat in hub.probe_store.cats]
    overlaps = []
    print('Calculating...')
    try:
        for r in results:
            overlap = r.get()
            overlaps.append(overlap)
        pool.close()
    except KeyboardInterrupt:
        pool.close()
        raise SystemExit('Interrupt occurred during multiprocessing. Closed worker pool.')
    print('mean overlap={:,}'.format(sum(overlaps) / len(overlaps)))
    print('with max_distance={}'.format(MAX_DISTANCE))
    print()

