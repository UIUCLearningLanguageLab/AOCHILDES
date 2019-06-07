import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.sparse import linalg as slinalg
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from childeshub.hub import Hub


"""
are nouns distributionally more dissimilar to verbs in partition 2 vs 1? 
if so, this would lend evidence to the idea that p2 induces syntactic bias (it is more important to learn
the difference between nous and verbs than differences between semantic categories. 
"""

SANITY_CHECK = False
HUB_MODE = 'sem'
NGRAM_SIZE = 6
BINARY = False
NUM_PCS = 512
Y_MAX = 0.35

FREQ_THR = 100


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


filtered_nouns = set([noun for noun in hub.nouns
                      if hub.train_terms.term_freq_dict[noun] > FREQ_THR]
                     + list(hub.probe_store.types))
filtered_verbs = set([verb for verb in hub.verbs
                      if hub.train_terms.term_freq_dict[verb] > FREQ_THR])
filtered_adjs = set([verb for verb in hub.adjectives
                     if hub.train_terms.term_freq_dict[verb] > FREQ_THR])
filtered_preps = set([prep for prep in hub.prepositions
                      if hub.train_terms.term_freq_dict[prep] > FREQ_THR])
filtered_conjs = set([conj for conj in hub.conjunctions
                      if hub.train_terms.term_freq_dict[conj] > FREQ_THR])
filtered_dets = set([det for det in hub.determiners
                     if hub.train_terms.term_freq_dict[det] > FREQ_THR])
filtered_pros = set([pro for pro in hub.pronouns
                     if hub.train_terms.term_freq_dict[pro] > FREQ_THR])
filtered_ints = set([pro for pro in hub.interjections
                     if hub.train_terms.term_freq_dict[pro] > FREQ_THR])


print('nouns')
print(filtered_nouns)
print('verbs')
print(filtered_verbs)
print('adjectives')
print(filtered_adjs)
print('interjections')
print(filtered_ints)

# make in_out_corr_mats
label1 = 'tokens between\n{:,} & {:,}'.format(START1, END1)
label2 = 'tokens between\n{:,} & {:,}'.format(START2, END2)
in_out_corr_mat1, types1 = make_in_out_corr_mat(START1, END1)
in_out_corr_mat2, types2 = make_in_out_corr_mat(START2, END2)


noun_verb_sims = []
noun_adj_sims = []
noun_prep_sims = []
noun_conj_sims = []
noun_det_sims = []
noun_pro_sims = []
noun_int_sims = []
for mat, types in [(in_out_corr_mat1.asfptype(), types1),
                   (in_out_corr_mat2.asfptype(), types2)]:
    print('Computing singular vectors ...')
    # compute  representations
    normalized = normalize(mat, axis=1, norm='l1', copy=False)
    u, s, v = slinalg.svds(normalized, k=NUM_PCS, return_singular_vectors='u')
    noun_bool_ids = [True if t in filtered_nouns else False for t in types]
    verb_bool_ids = [True if t in filtered_verbs else False for t in types]
    adj_bool_ids = [True if t in filtered_adjs else False for t in types]
    prep_bool_ids = [True if t in filtered_preps else False for t in types]
    conj_bool_ids = [True if t in filtered_conjs else False for t in types]
    det_bool_ids = [True if t in filtered_dets else False for t in types]
    pro_bool_ids = [True if t in filtered_pros else False for t in types]
    int_bool_ids = [True if t in filtered_ints else False for t in types]
    noun_reps = u[noun_bool_ids]
    verb_reps = u[verb_bool_ids]
    adj_reps = u[adj_bool_ids]
    prep_reps = u[prep_bool_ids]
    conj_reps = u[conj_bool_ids]
    det_reps = u[det_bool_ids]
    pro_reps = u[pro_bool_ids]
    int_reps = u[int_bool_ids]
    # average over representations
    avg_noun_rep = noun_reps.mean(0, keepdims=True)
    avg_verb_rep = verb_reps.mean(0, keepdims=True)
    avg_adj_rep = adj_reps.mean(0, keepdims=True)
    avg_prep_rep = prep_reps.mean(0, keepdims=True)
    avg_conj_rep = conj_reps.mean(0, keepdims=True)
    avg_det_rep = det_reps.mean(0, keepdims=True)
    avg_pro_rep = pro_reps.mean(0, keepdims=True)
    avg_int_rep = int_reps.mean(0, keepdims=True)
    # cosine similarity
    noun_verb_sim = cosine_similarity(avg_noun_rep, avg_verb_rep).mean()
    noun_adj_sim = cosine_similarity(avg_noun_rep, avg_adj_rep).mean()
    noun_prep_sim = cosine_similarity(avg_noun_rep, avg_prep_rep).mean()
    noun_conj_sim = cosine_similarity(avg_noun_rep, avg_conj_rep).mean()
    noun_det_sim = cosine_similarity(avg_noun_rep, avg_det_rep).mean()
    noun_pro_sim = cosine_similarity(avg_noun_rep, avg_pro_rep).mean()
    noun_int_sim = cosine_similarity(avg_noun_rep, avg_int_rep).mean()
    # collect
    noun_verb_sims.append(noun_verb_sim)
    noun_adj_sims.append(noun_adj_sim)
    noun_prep_sims.append(noun_prep_sim)
    noun_conj_sims.append(noun_conj_sim)
    noun_det_sims.append(noun_det_sim)
    noun_pro_sims.append(noun_pro_sim)
    noun_int_sims.append(noun_int_sim)
    print('noun_verb_sim={}'.format(noun_verb_sim))
    print('noun_adj_sim={}'.format(noun_adj_sim))
    print('noun_prep_sim={}'.format(noun_prep_sim))
    print('noun_conj_sim={}'.format(noun_conj_sim))
    print('noun_det_sim={}'.format(noun_det_sim))
    print('noun_pro_sim={}'.format(noun_pro_sim))
    print('noun_int_sim={}'.format(noun_int_sim))
    print('------------------------------------------------------')


# fig
x = [0, 1]
fig, ax = plt.subplots(dpi=None, figsize=(5, 5))
plt.title('Distributional similarity of nouns & other POS classes\nn-gram model with size={} and num PCs={}'.format(
    NGRAM_SIZE, NUM_PCS))
ax.set_ylabel('Cosine Similarity')
ax.set_xlabel('Partition')
ax.set_ylim([0, Y_MAX])
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.plot(x, noun_verb_sims, label='verb')
ax.plot(x, noun_adj_sims, label='adjective')
ax.plot(x, noun_prep_sims, label='preposition')
ax.plot(x, noun_conj_sims, label='conjunction')
ax.plot(x, noun_det_sims, label='determiner')
ax.plot(x, noun_pro_sims, label='pronoun')
ax.plot(x, noun_int_sims, label='interjection')
plt.legend()
fig.tight_layout()
plt.show()