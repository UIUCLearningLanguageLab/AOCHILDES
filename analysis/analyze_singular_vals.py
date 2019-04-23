import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as slinalg
from scipy import sparse

from childeshub.hub import Hub

SANITY_CHECK = False

HUB_MODE = 'sem'
NUM_PCS = 128
NGRAM_SIZE = 3
BINARY = False

LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (10, 4)
DPI = None

hub = Hub(mode=HUB_MODE)

START1, END1 = 0, hub.midpoint_loc // 1
START2, END2 = hub.train_terms.num_tokens - END1, hub.train_terms.num_tokens

CATS = [] or hub.probe_store.cats  # ['bug', 'dessert', 'body', 'bug']

adj_alpha = 0.05 / NUM_PCS


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
    mat_loc2freq = {}  # to keep track of number of ngram & type co-occurence
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


def plot_comparison(y1, y2):
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title('data represented with ngram-size={}'.format(NGRAM_SIZE), fontsize=AX_FONTSIZE)
    ax.set_ylabel('singular value', fontsize=AX_FONTSIZE)
    ax.set_xlabel('Principal Component #', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    ax.plot(y1, label=label1, linewidth=2)
    ax.plot(y2, label=label2, linewidth=2)
    ax.legend(loc='upper left', frameon=False, fontsize=LEG_FONTSIZE)
    plt.tight_layout()
    plt.show()


# make in_out_corr_mats
label1 = 'tokens between\n{:,} & {:,}'.format(START1, END1)
label2 = 'tokens between\n{:,} & {:,}'.format(START2, END2)
in_out_corr_mat1, types1 = make_in_out_corr_mat(START1, END1)
in_out_corr_mat2, types2 = make_in_out_corr_mat(START2, END2)


# pca1
print('Fitting PCA 1 ...')
sparse_in_out_corr_mat1 = sparse.csr_matrix(in_out_corr_mat1).asfptype()
_, s1, _ = slinalg.svds(sparse_in_out_corr_mat1, k=NUM_PCS)

# pca2
print('Fitting PCA 2 ...')
sparse_in_out_corr_mat2 = sparse.csr_matrix(in_out_corr_mat2).asfptype()
_, s2, _ = slinalg.svds(sparse_in_out_corr_mat2, k=NUM_PCS)


# analyze s
y1 = []
y2 = []
for s, y in [(s1, y1), (s2, y2)]:
    for row in s[:NUM_PCS]:
        sing_val = row.sum()
        print(sing_val)
        y.append(sing_val)
    print()

plot_comparison(y1, y2)