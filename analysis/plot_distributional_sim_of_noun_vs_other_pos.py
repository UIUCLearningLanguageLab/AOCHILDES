import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.sparse import linalg as slinalg
from sklearn.metrics.pairwise import cosine_similarity

from childeshub.hub import Hub


"""
are nouns distributionally more dissimilar to verbs in partition 2 vs 1? 
if so, this would lend evidence to the idea that p2 induces syntactic bias (it is more important to learn
the difference between nous and verbs than differences between semantic categories. 
"""

HUB_MODE = 'sem'
WINDOW_SIZE = 2
NUM_PCS = 512
Y_MAX = 0.5

FREQ_THR = 100
VERBOSE = False


hub = Hub(mode=HUB_MODE)

START1, END1 = 0, hub.midpoint_loc // 1
START2, END2 = hub.train_terms.num_tokens - END1, hub.train_terms.num_tokens


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

if VERBOSE:
    print('nouns')
    print(filtered_nouns)
    print('verbs')
    print(filtered_verbs)
    print('adjectives')
    print(filtered_adjs)
    print('interjections')
    print(filtered_ints)

# make term_by_window_co_occurrence_mats
start1, end1 = 0, hub.midpoint_loc // 1
start2, end2 = hub.train_terms.num_tokens - end1, hub.train_terms.num_tokens
label1 = 'partition 1' or 'tokens between\n{:,} & {:,}'.format(start1, end1)
label2 = 'partition 2' or 'tokens between\n{:,} & {:,}'.format(start2, end2)
tw_mat1, xws1, yws1 = hub.make_term_by_window_co_occurrence_mat(start=start1, end=end1, window_size=WINDOW_SIZE)
tw_mat2, xws2, yws2 = hub.make_term_by_window_co_occurrence_mat(start=start2, end=end2, window_size=WINDOW_SIZE)


noun_verb_sims = []
noun_adj_sims = []
noun_prep_sims = []
noun_conj_sims = []
noun_det_sims = []
noun_pro_sims = []
noun_int_sims = []
for mat, xws in [(tw_mat1.asfptype(), xws1),
                 (tw_mat2.asfptype(), xws2)]:
    print('Computing singular vectors ...')
    # compute  representations
    mat = mat.T  # transpose to do SVD, x-words now index rows
    normalized = normalize(mat, axis=1, norm='l1', copy=False)  # axis=0 to normalize features else samples
    noun_bool_ids = [True if t in filtered_nouns else False for t in xws]
    verb_bool_ids = [True if t in filtered_verbs else False for t in xws]
    adj_bool_ids = [True if t in filtered_adjs else False for t in xws]
    prep_bool_ids = [True if t in filtered_preps else False for t in xws]
    conj_bool_ids = [True if t in filtered_conjs else False for t in xws]
    det_bool_ids = [True if t in filtered_dets else False for t in xws]
    pro_bool_ids = [True if t in filtered_pros else False for t in xws]
    int_bool_ids = [True if t in filtered_ints else False for t in xws]
    if NUM_PCS is not None:
        u, s, v = slinalg.svds(normalized, k=NUM_PCS, return_singular_vectors='u')
        avg_noun_rep = u[noun_bool_ids].max(0, keepdims=True)
        avg_verb_rep = u[verb_bool_ids].max(0, keepdims=True)
        avg_adj_rep = u[adj_bool_ids].max(0, keepdims=True)
        avg_prep_rep = u[prep_bool_ids].max(0, keepdims=True)
        avg_conj_rep = u[conj_bool_ids].max(0, keepdims=True)
        avg_det_rep = u[det_bool_ids].max(0, keepdims=True)
        avg_pro_rep = u[pro_bool_ids].max(0, keepdims=True)
        avg_int_rep = u[int_bool_ids].max(0, keepdims=True)
    else:
        avg_noun_rep = normalized.todense()[noun_bool_ids].max(0)
        avg_verb_rep = normalized.todense()[verb_bool_ids].max(0)
        avg_adj_rep = normalized.todense()[adj_bool_ids].max(0)
        avg_prep_rep = normalized.todense()[prep_bool_ids].max(0)
        avg_conj_rep = normalized.todense()[conj_bool_ids].max(0)
        avg_det_rep = normalized.todense()[det_bool_ids].max(0)
        avg_pro_rep = normalized.todense()[pro_bool_ids].max(0)
        avg_int_rep = normalized.todense()[int_bool_ids].max(0)
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
    WINDOW_SIZE, NUM_PCS))
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