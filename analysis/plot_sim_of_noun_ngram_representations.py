import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.sparse import linalg as slinalg
from scipy import sparse

from childeshub.hub import Hub


"""
are nouns distributionally more similar in partition 2 vs 1? if so, this would lend evidence to the idea that
p2 reinforces that probes are nouns, whereas in p1 this constraint is reduced 
(thereby improving speed of semantic differentiation) 
"""

SANITY_CHECK = False

HUB_MODE = 'sem'
NGRAM_SIZE = 6
BINARY = False
NUM_PCS = 512  # set to None to skip SVD  - memory error when using NGRAM_SIZE > 3

PROBES = False
NOUN_FREQ_THR = 100


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


if not PROBES:
    filtered_nouns = set([noun for noun in hub.nouns
                          if hub.train_terms.term_freq_dict[noun] > NOUN_FREQ_THR]
                         + list(hub.probe_store.types))
else:
    filtered_nouns = set(hub.probe_store.types)


# make in_out_corr_mats
label1 = 'tokens between\n{:,} & {:,}'.format(START1, END1)
label2 = 'tokens between\n{:,} & {:,}'.format(START2, END2)
in_out_corr_mat1, types1 = make_in_out_corr_mat(START1, END1)
in_out_corr_mat2, types2 = make_in_out_corr_mat(START2, END2)


noun_sims = []
for mat, types in [(in_out_corr_mat1.asfptype(), types1),
                   (in_out_corr_mat2.asfptype(), types2)]:
    print('Computing singular vectors ...')
    # compute noun representations
    normalized = normalize(mat, axis=1, norm='l1', copy=False)
    bool_ids = [True if t in filtered_nouns else False for t in types]
    if NUM_PCS is not None:
        u, s, v = slinalg.svds(normalized, k=NUM_PCS, return_singular_vectors='u')
        noun_reps = u[bool_ids]
    else:
        noun_reps = normalized.todense()[bool_ids]
    # collect sim
    noun_sim = np.corrcoef(noun_reps, rowvar=True).mean().item()  # rowvar is correct
    noun_sims.append(noun_sim)
    print('noun_sim={}'.format(noun_sim))
    print('------------------------------------------------------')


# fig
x = [0, 1]
fig, ax = plt.subplots(dpi=None, figsize=(5, 5))
plt.title('Distributional similarity of {}\nn-gram model with size={} and num PCs={}'.format(
    'probes' if PROBES else 'nouns', NGRAM_SIZE, NUM_PCS))
ax.set_ylabel('Similarity')
ax.set_xlabel('Partition')
ax.set_ylim([0, np.max(noun_sims) + 0.01])
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.plot(x, noun_sims)
fig.tight_layout()
plt.show()