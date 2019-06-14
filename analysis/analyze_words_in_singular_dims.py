import numpy as np
from scipy.sparse import linalg as slinalg
from scipy import sparse
from sklearn.preprocessing import normalize

from childeshub.hub import Hub

SANITY_CHECK = False

HUB_MODE = 'sem'
NGRAM_SIZE = 6
BINARY = False
NUM_PCS = 10
MIN_FREQ = 30


hub = Hub(mode=HUB_MODE)

START1, END1 = 0, hub.midpoint_loc // 1
START2, END2 = hub.train_terms.num_tokens - END1, hub.train_terms.num_tokens


# make in_out_corr_mats
label1 = 'tokens between\n{:,} & {:,}'.format(START1, END1)
label2 = 'tokens between\n{:,} & {:,}'.format(START2, END2)
in_out_corr_mat1, types1 = hub.make_term_by_window_co_occurrence_mat(START1, END1)
in_out_corr_mat2, types2 = hub.make_term_by_window_co_occurrence_mat(START2, END2)
in_out_corr_mat_all, types_all = hub.make_term_by_window_co_occurrence_mat(START1, END2)  # using both partitions

# do svd
for mat, types, name in [(in_out_corr_mat1.asfptype(), types1, 'partition 1'),
                   (in_out_corr_mat2.asfptype(), types2, 'partition 2'),
                   (in_out_corr_mat_all.asfptype(), types_all, 'partition 1+2')]:
    print('Fitting SVD on {}...'.format(name))
    normalized = normalize(mat, axis=1, norm='l2', copy=False)
    u, s, _ = slinalg.svds(normalized, k=NUM_PCS, return_singular_vectors=True)  # s is not 2D
    #
    for pc_id in np.arange(NUM_PCS):
        print('Singular Dim={} s={}'.format(NUM_PCS - pc_id, s[pc_id]))
        sorted_ids = np.argsort(u[:, pc_id])
        print([types[i] for i in sorted_ids[:20] if hub.train_terms.term_freq_dict[types[i]] > MIN_FREQ])
        print([types[i] for i in sorted_ids[-20:] if hub.train_terms.term_freq_dict[types[i]] > MIN_FREQ])
        print()
    print('------------------------------------------------')

