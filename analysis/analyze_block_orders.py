import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from childeshub.hub import Hub


CORPUS_NAME = 'childes-20180319'
HUB_MODE = 'sem'
NUM_PARTS = 256
DPI = 196
ANNOT_SIZE = 10  # 6
NGRAM_SIZES = [1, 2, 3, 4, 5, 6]

ORDER_FEATURES = ['age',  # need to have age in there
                  'conjunction',
                  'preposition',
                  'pronoun',
                  'adverb',
                  'verb',
                  'adverb+conjunction',
                  'adverb+preposition',
                  'verb+conjunction',
                  'verb+preposition',
                  'pronoun+conjunction',
                  'pronoun+preposition']

num_order_features = len(ORDER_FEATURES)
hub = Hub(mode=HUB_MODE, num_parts=NUM_PARTS, corpus_name=CORPUS_NAME, part_order='inc_age')


def make_features_mat(parts):
    print('Making features_mat')
    result = np.zeros((hub.num_parts, num_order_features))
    for col_id, order_feature in enumerate(ORDER_FEATURES):
        print('Calculating part stats {}/{} using "{}"...'.format(col_id + 1, num_order_features, order_feature))
        part_id_sort_stat_dict = hub.calc_part_id_sort_stat_dict(parts, order_feature)
        col = [part_id_sort_stat_dict[part_id] for part_id in range(hub.num_parts)]
        result[:, col_id] = col
    return result


# stats
ao_features_mat = make_features_mat(hub.reordered_partitions)
rho_mat, p_mat = spearmanr(ao_features_mat)
print(p_mat < 0.05 / ao_features_mat.size)

# fig
fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)
sns.heatmap(rho_mat, ax=ax, square=True, annot=True,
            annot_kws={"size": ANNOT_SIZE}, cbar_kws={"shrink": .5},
            cmap='jet', vmin=-1, vmax=1)
cbar = ax.collections[0].colorbar
cbar.set_label('Spearman Rank Correlation (rho)')
ax.set_yticklabels(ORDER_FEATURES, rotation=0)
ax.set_xticklabels(ORDER_FEATURES, rotation=90)
fig.tight_layout()
plt.show()
