import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import pandas as pd

from childeshub.hub import Hub

CORPUS_NAME = 'childes-20180319'
HUB_MODE = 'sem'
NUM_PARTS = 256

DPI = 196
ANNOT_SIZE = 5
FONT_SCALE = 1.4
FIGSIZE = (16, 16)

BLOCK_ORDERS = ['dec_punctuation',
                'inc_1-gram',
                'inc_3-gram',
                'inc_6-gram',
                'inc_preposition',
                'inc_conjunction',
                'inc_entropy']
BLOCK_ORDERS += ['dec_noun+punctuation',
                 'inc_verb+conjunction',
                 'inc_verb+preposition',
                 'inc_adverb+conjunction',
                 'inc_adverb+preposition']
BLOCK_ORDERS += ['dec_noun',
                 'inc_verb',
                 'inc_adverb',
                 'dec_determiner',
                 'inc_pronoun']

order_features = [bo.split('_')[-1] for bo in BLOCK_ORDERS]
num_order_features = len(order_features)
num_part_orders = len(BLOCK_ORDERS)


def make_features_mat(parts):
    print('Making features_mat')
    result = np.zeros((hub.num_parts, num_order_features))
    for col_id, order_feature in enumerate(order_features):
        print('Calculating part stats {}/{} using "{}"...'.format(col_id + 1, num_order_features, order_feature))
        part_id_sort_stat_dict = hub.calc_part_id_sort_stat_dict(parts, order_feature)
        col = [part_id_sort_stat_dict[part_id] for part_id in range(hub.num_parts)]
        result[:, col_id] = col
    return result


hub = Hub(mode=HUB_MODE, num_parts=NUM_PARTS, corpus_name=CORPUS_NAME, part_order='inc_age')
ao_feature_mat = make_features_mat(hub.reordered_partitions)

# make corr_mat
corr_mat = np.zeros((num_part_orders, num_order_features))  # TODO paralellize algorithm?
for row_id, part_order in enumerate(BLOCK_ORDERS):
    print(part_order, '\n/////////////////////////////////////////')
    reordered_partitions = hub.reorder_parts(part_order)
    bo_feature_mat = make_features_mat(reordered_partitions)
    row = [spearmanr(ao_feats, bo_feats)[0]  # rho
           for ao_feats, bo_feats in zip(ao_feature_mat.T, bo_feature_mat.T)]
    corr_mat[row_id, :] = row
    print(corr_mat)


# fig
fig, ax = plt.subplots(figsize=(10, 10), dpi=DPI)
sns.heatmap(corr_mat, ax=ax, annot=True, square=True,
            annot_kws={"size": ANNOT_SIZE}, cbar_kws={"shrink": .5},
            cmap='jet', vmin=-1, vmax=1)
cbar = ax.collections[0].colorbar
cbar.set_label('Spearman Rank Correlation (rho)')
ax.set_xticklabels(order_features, rotation=90)
ax.set_yticklabels(BLOCK_ORDERS, rotation=0)
ax.set_xlabel('Features')
ax.set_ylabel('Order')
fig.tight_layout()
plt.show()


# fig - clustered
df = pd.DataFrame(corr_mat, columns=order_features, index=BLOCK_ORDERS)
print('which part_orders have best correlation?')
print(df.mean(axis=1).sort_values())
df.drop([f for f in order_features if '+' in f], axis=1, inplace=True)
# df.drop(['inc_probes-spread-stdlocs', 'inc_probes-context-entropy-1-left'], axis=0, inplace=True)
sns.set(font_scale=FONT_SCALE)
cm = sns.clustermap(df, metric='cosine', method='average', square=False,
                    cmap=plt.cm.jet, figsize=FIGSIZE)
# remove cbar
# cm.cax.set_visible(False)
# reposition box to fit all labels
left_x = 0.1
shrink_prop_y = 0.15
for ax in [cm.ax_row_dendrogram, cm.ax_heatmap]:
    box = ax.get_position()
    ax.set_position([box.x0 - left_x,
                     box.y0 + (box.height * shrink_prop_y),
                     box.width,
                     box.height * (1 - shrink_prop_y)])
ax = cm.ax_col_dendrogram
box = ax.get_position()
ax.set_position([box.x0 - left_x,
                 box.y0,
                 box.width,
                 box.height])
plt.show()