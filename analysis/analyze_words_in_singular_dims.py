import numpy as np
from scipy.sparse import linalg as slinalg
from sklearn.preprocessing import normalize

from childeshub.hub import Hub

SANITY_CHECK = False

HUB_MODE = 'sem'
WINDOW_SIZE = 6
NUM_SVS = 10
MIN_FREQ = 30


hub = Hub(mode=HUB_MODE)

# make term_by_window_co_occurrence_mats
start1, end1 = 0, hub.midpoint_loc // 1
start2, end2 = hub.train_terms.num_tokens - end1, hub.train_terms.num_tokens
label1 = 'partition 1' or 'tokens between\n{:,} & {:,}'.format(start1, end1)
label2 = 'partition 2' or 'tokens between\n{:,} & {:,}'.format(start2, end2)
tw_mat1, xws1, yws1 = hub.make_term_by_window_co_occurrence_mat(start=start1, end=end1, window_size=WINDOW_SIZE)
tw_mat2, xws2, yws2 = hub.make_term_by_window_co_occurrence_mat(start=start2, end=end2, window_size=WINDOW_SIZE)
tw_mat3, xws3, yws3 = hub.make_term_by_window_co_occurrence_mat(start=start2, end=end2, window_size=WINDOW_SIZE)


# do svd
for mat, types, name in [(tw_mat1.asfptype(), xws1, 'partition 1'),
                         (tw_mat2.asfptype(), xws2, 'partition 2'),
                         (tw_mat3.asfptype(), xws3, 'partition 1+2')]:
    print('Fitting SVD on {}...'.format(name))
    mat = mat.T  # transpose to do SVD, x-words now index rows
    normalized = normalize(mat, axis=1, norm='l2', copy=False)
    u, s, _ = slinalg.svds(normalized, k=NUM_SVS, return_singular_vectors=True)  # s is not 2D
    #
    for pc_id in np.arange(NUM_SVS):
        print('Singular Dim={} s={}'.format(NUM_SVS - pc_id, s[pc_id]))
        sorted_ids = np.argsort(u[:, pc_id])
        print([types[i] for i in sorted_ids[:20] if hub.train_terms.term_freq_dict[types[i]] > MIN_FREQ])
        print([types[i] for i in sorted_ids[-20:] if hub.train_terms.term_freq_dict[types[i]] > MIN_FREQ])
        print()
    print('------------------------------------------------')

