import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from childeshub import config

CORPUS_NAME = 'childes-20180319'
HUB_MODE = 'sem'
NUM_PARTS = 256
DPI = 196
NGRAM_SIZES = [1, 2, 3, 4, 5, 6]  # this should be same as used to generate rho_mat
THRESHOLD = 0.6

# make corr
rho_mat = np.abs(np.load('rho_mat.npy'))
pos_list = sorted(config.Terms.pos2tags.keys())
names = ['age',
         'lexical diversity'] + \
        ['num {}-grams'.format(n) for n in NGRAM_SIZES] + \
        ['num {}s'.format(pos) for pos in pos_list]
corr = pd.DataFrame(rho_mat, columns=names, index=names)
corr.drop(['num 2-grams', 'num 4-grams', 'num 5-grams'], axis=1, inplace=True)
corr.drop(['num 2-grams', 'num 4-grams', 'num 5-grams'], axis=0, inplace=True)

corr.index = [name.replace('num ', '') for name in corr.index]
corr.columns = [name.replace('num ', '') for name in corr.columns]

# make links and threshold
links = corr.stack().reset_index()
links.columns = ['var1', 'var2','value']
links_filtered = links.loc[(links['value'] > THRESHOLD) & (links['var1'] != links['var2']) ]

# plot graph
fig, ax = plt.subplots(dpi=DPI)
G = nx.DiGraph()
G.add_weighted_edges_from([tuple(link) for link in links_filtered.values])
pos = nx.kamada_kawai_layout(G)
nx.draw(G,
        ax=ax,
        pos=pos,
        with_labels=True,
        node_color='orange',
        node_size=200,
        edge_color=links_filtered['value'].tolist(),
        edge_cmap=plt.cm.Blues,
        linewidths=1,
        font_size=8)
plt.title('Correlation Matrix Network with link threshold={}'.format(THRESHOLD))
plt.show()