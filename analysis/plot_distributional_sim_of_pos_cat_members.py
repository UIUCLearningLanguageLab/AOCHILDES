import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.sparse import linalg as slinalg


from childeshub.hub import Hub


"""
how similar are term-by-window representations of members of a POS category?
"""


HUB_MODE = 'sem'
WINDOW_SIZE = 7
NUM_PCS = 512  # set to None to skip SVD  - memory error when using WINDOW_SIZE > 3
FONTSIZE = 16

POS = 'nouns'


hub = Hub(mode=HUB_MODE)

filtered_pos_members = set([pos_member for pos_member in getattr(hub, POS)])
print('there are {} {}'.format(len(filtered_pos_members), POS))

# make term_by_window_co_occurrence_mats
start1, end1 = 0, hub.midpoint_loc // 1
start2, end2 = hub.train_terms.num_tokens - end1, hub.train_terms.num_tokens
label1 = 'partition 1' or 'tokens between\n{:,} & {:,}'.format(start1, end1)
label2 = 'partition 2' or 'tokens between\n{:,} & {:,}'.format(start2, end2)
tw_mat1, xws1, yws1 = hub.make_term_by_window_co_occurrence_mat(start=start1, end=end1, window_size=WINDOW_SIZE)
tw_mat2, xws2, yws2 = hub.make_term_by_window_co_occurrence_mat(start=start2, end=end2, window_size=WINDOW_SIZE)


pos_member_sims = []
for mat, xws in [(tw_mat1.asfptype(), xws1),
                 (tw_mat2.asfptype(), xws2)]:
    print('Computing SVD ...')
    # compute pos_member representations
    mat = mat.T  # transpose to do SVD, x-words now index rows
    normalized = normalize(mat, axis=1, norm='l1', copy=False)
    bool_ids = [True if t in filtered_pos_members else False for t in xws]
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
    POS, WINDOW_SIZE, NUM_PCS), fontsize=FONTSIZE)
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