import numpy as np
from scipy.sparse import linalg as slinalg
from scipy import sparse
from sklearn.preprocessing import normalize

from childeshub.hub import Hub

SANITY_CHECK = False

HUB_MODE = 'sem'
NGRAM_SIZE = 1
BINARY = False
NUM_PCS = 3
MIN_FREQ = 30


hub = Hub(mode=HUB_MODE)

START1, END1 = 0, hub.midpoint_loc // 1
START2, END2 = hub.train_terms.num_tokens - END1, hub.train_terms.num_tokens


def make_in_out_corr_mat(start, end):
    print('Making in_out_corr_mat with start={} and end={}'.format(start, end))
    tokens = hub.train_terms.tokens[start:end]
    # types
    types = sorted(set(tokens))
    num_types = len(types)
    type2id = {t: n for n, t in enumerate(types)}
    # ngrams
    ngrams = hub.get_ngrams(NGRAM_SIZE, tokens)
    ngram_types = sorted(set(ngrams))
    num_ngram_types = len(ngram_types)
    ngram2id = {t: n for n, t in enumerate(ngram_types)}
    # make sparse matrix (types in rows, ngrams in cols)
    shape = (num_types, num_ngram_types)
    print('Making In-Out matrix with shape={}...'.format(shape))
    data = []
    row_ids = []
    cold_ids = []
    mat_loc2freq = {}  # to keep track of number of ngram & type co-occurrence
    for n, ngram in enumerate(ngrams[:-NGRAM_SIZE]):
        # row_id + col_id
        col_id = ngram2id[ngram]
        next_ngram = ngrams[n + 1]
        next_type = next_ngram[-1]
        row_id = type2id[next_type]
        # freq
        try:
            freq = mat_loc2freq[(row_id, col_id)]
        except KeyError:
            mat_loc2freq[(row_id, col_id)] = 1
            freq = 1
        else:
            mat_loc2freq[(row_id, col_id)] += 1
        # collect
        row_ids.append(row_id)
        cold_ids.append(col_id)
        data.append(1 if BINARY else freq)
    # make sparse matrix once (updating it is expensive)
    res = sparse.csr_matrix((data, (row_ids, cold_ids)), shape=(num_types, num_ngram_types))
    #
    if SANITY_CHECK:
        for term_id in range(num_types):
            print('----------------------------------')
            print(types[term_id])
            print('----------------------------------')
            for ngram_id, freq in enumerate(np.squeeze(res[term_id].toarray())):
                print(ngram_types[ngram_id], freq) if freq != 0 else None
        raise SystemExit
    return res, types


# make in_out_corr_mats
label1 = 'tokens between\n{:,} & {:,}'.format(START1, END1)
label2 = 'tokens between\n{:,} & {:,}'.format(START2, END2)
in_out_corr_mat1, types1 = make_in_out_corr_mat(START1, END2)  # TODO using both partitions
# in_out_corr_mat2, types2 = make_in_out_corr_mat(START2, END2)


# do svd
for mat, types in [(in_out_corr_mat1.asfptype(), types1)]:
    print('Fitting PCA ...')
    normalized = normalize(mat, axis=1, norm='l2', copy=False)
    u, s, _ = slinalg.svds(normalized, k=NUM_PCS, return_singular_vectors=True)  # s is not 2D

    # TODO how to get explained_variance?
    print(s)
    print([(u[:, i] ** 2).sum() / (len(types) - 1) for i in range(NUM_PCS)])
    tot_var = np.var(mat.todense())
    print(tot_var)
    print([si / tot_var for si in s])

    for pc_id in range(NUM_PCS):
        print('PC={} s={}'.format(pc_id + 1, s[pc_id]))
        sorted_ids = np.argsort(u[:, pc_id])
        print([types[i] for i in sorted_ids[:20] if hub.train_terms.term_freq_dict[types[i]] > MIN_FREQ])
        print([types[i] for i in sorted_ids[-20:] if hub.train_terms.term_freq_dict[types[i]] > MIN_FREQ])

