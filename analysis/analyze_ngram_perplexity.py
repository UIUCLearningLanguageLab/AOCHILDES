import kenlm
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle
from pathlib import Path

from childeshub import config
from childeshub.hub import Hub

NGRAM_SIZES = [4, 5, 6]  # 2 and 3 expand y axis too much
NUM_TYPES_LIST = [4096, 23 * 1000]
LMPLZ_PATH = '/home/ph/starting_small/kenlm/bin/lmplz'  # these binaries must be installed by user
BINARIZE_PATH = '/home/ph/starting_small/kenlm/bin/build_binary'
MODEL_EXTENSION = 'klm'  # 'klm' is faster than 'arpa'
CORPUS_NAME = 'childes-20180319'
DELETE_PREVIOUS_MODELS = True

NGRAMS_MODEL_DIR = 'home/ph/ngram_models'

for f in Path(NGRAMS_MODEL_DIR).glob('*'):
    f.unlink()


def calc_pps(str1, str2):
    result = []
    for s1, s2 in [(str1, str1), (str2, str2)]:
        # train n-gram model
        p_in = config.Dirs.src / 'rnnlab' / 'ngram_models' / 'temp.txt'
        p_in.write_text(s1)
        p = Popen([LMPLZ_PATH, '-o', str(ngram_size)], stdin=p_in.open(), stdout=PIPE)
        p_in.unlink()
        # save model
        p_out = Path(NGRAMS_MODEL_DIR) / '{}_{}-grams.arpa'.format(CORPUS_NAME, ngram_size)
        if not p_out.exists():
            p_out.touch()
        arpa_file_bytes = p.stdout.read()
        p_out.write_text(arpa_file_bytes.decode())
        # binarize model
        klm_file_path = str(p_out).rstrip('arpa') + 'klm'
        p = Popen([BINARIZE_PATH, str(p_out), klm_file_path])
        p.wait()
        # load model
        print('Computing perplexity using {}-gram model...'.format(ngram_size))
        model = kenlm.Model(klm_file_path)
        # score
        pp = model.perplexity(s2)
        result.append(pp)
    print(result)
    return result


# xys_list
xys_list = []
for num_types in NUM_TYPES_LIST:
    hub = Hub(corpus_name=CORPUS_NAME, num_types=num_types)
    xys = []
    for ngram_size in NGRAM_SIZES:
        y = calc_pps(' '.join(hub.first_half_tokens), ' '.join(hub.second_half_tokens))
        xys.append((y, ngram_size))
    xys_list.append(xys)

# fig
bar_width = 0.35
num_vocab_sizes = len(NUM_TYPES_LIST)
fig, axarr = plt.subplots(num_vocab_sizes, 1, dpi=192)
palette = cycle(sns.color_palette("hls", 2)[::-1])
for title, ax, xys in zip(['Reduced Vocabulary', 'Full Vocabulary'], axarr, xys_list):
    ax.set_title(title)
    ax.set_ylabel('Perplexity')
    ax.set_xlabel('N-gram size')
    # ax.set_ylim([0, 30])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.set_xticks(np.array(NGRAM_SIZES) + bar_width / 2)
    ax.set_xticklabels(NGRAM_SIZES)
    # plot
    for n, (y, ngram_size) in enumerate(xys):
        x = np.array([ngram_size, ngram_size + bar_width])
        for x_single, y_single, c, label in zip(x, y, palette, ['partition 1', 'partition 2']):
            label = label if n == 0 else '_nolegend_'  # label only once
            ax.bar(x_single, y_single, bar_width, color=c, label=label)
plt.legend()
plt.tight_layout()
fig.show()

for title, xys in zip(['Reduced Vocabulary', 'Full Vocabulary'], xys_list):
    print(title)
    for y in xys:
        print(y)
