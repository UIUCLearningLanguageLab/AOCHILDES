import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.sparse import linalg as slinalg


from childeshub.hub import Hub


"""
how similar are n-gram representations of members of a POS category?
"""


HUB_MODE = 'sem'
NGRAM_SIZE = 7
BINARY = False
NUM_PCS = 512  # set to None to skip SVD  - memory error when using NGRAM_SIZE > 3
FONTSIZE = 16

FREQ_THR = 10

POS = 'nouns'


hub = Hub(mode=HUB_MODE)

START1, END1 = 0, hub.midpoint_loc // 1
START2, END2 = hub.train_terms.num_tokens - END1, hub.train_terms.num_tokens


filtered_pos_members = set([pos_member for pos_member in getattr(hub, POS)
                            if hub.train_terms.term_freq_dict[pos_member] > FREQ_THR])
print('there are {} {}'.format(len(filtered_pos_members), POS))


# make in_out_corr_mats
label1 = 'tokens between\n{:,} & {:,}'.format(START1, END1)
label2 = 'tokens between\n{:,} & {:,}'.format(START2, END2)
in_out_corr_mat1, types1 = hub.make_term_by_window_co_occurrence_mat(START1, END1)
in_out_corr_mat2, types2 = hub.make_term_by_window_co_occurrence_mat(START2, END2)


pos_member_sims = []
for mat, types in [(in_out_corr_mat1.asfptype(), types1),
                   (in_out_corr_mat2.asfptype(), types2)]:
    print('Computing singular vectors ...')
    # compute pos_member representations
    normalized = normalize(mat, axis=1, norm='l1', copy=False)
    bool_ids = [True if t in filtered_pos_members else False for t in types]
    if NUM_PCS is not None:
        u, s, v = slinalg.svds(normalized, k=NUM_PCS, return_singular_vectors='u')
        pos_member_reps = u[bool_ids]
    else:
        pos_member_reps = normalized.todense()[bool_ids]
    print('found {} {} representations'.format(len(pos_member_reps), POS))
    # collect sim
    pos_member_sim = np.corrcoef(pos_member_reps, rowvar=True).mean().item()  # rowvar is correct
    pos_member_sims.append(pos_member_sim)
    print('pos_member_sim={}'.format(pos_member_sim))
    print('------------------------------------------------------')


# fig
x = [0, 1]
fig, ax = plt.subplots(dpi=None, figsize=(8, 8))
plt.title('Distributional similarity of {}\nwindow size={} and number of SVD modes={}'.format(
    POS, NGRAM_SIZE, NUM_PCS), fontsize=FONTSIZE)
ax.set_ylabel('Cosine Similarity', fontsize=FONTSIZE)
ax.set_xlabel('Partition', fontsize=FONTSIZE)
ax.set_ylim([0, np.max(pos_member_sims) + 0.01])
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.plot(x, pos_member_sims)
fig.tight_layout()
plt.show()