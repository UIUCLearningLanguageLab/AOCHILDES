import matplotlib.pyplot as plt

from childeshub.hub import Hub
from matplotlib.ticker import FormatStrFormatter

HUB_MODE = 'sem'

hub = Hub(mode=HUB_MODE)

AX_FONTSIZE = 8
LEG_FONTSIZE = 6
FIGSIZE = (3.2, 2.2)
DPI = 192
IS_LOG = True
WSPACE = 0.0
HSPACE = 0.0
WPAD = 0.0
HPAD = 0.0
PAD = 0.2
LW = 0.5

# xys
ys = [hub.part_entropies,
      hub.make_sentence_length_stat(hub.reordered_tokens, is_avg=True),
      hub.make_sentence_length_stat(hub.reordered_tokens, is_avg=False)]

# fig
ylabels = ['Shannon\nEntropy', 'Mean Utterance\nLength', 'Std Utterance\nLength']
fig, axs = plt.subplots(3, 1, dpi=DPI, figsize=FIGSIZE)
for ax, ylabel, y in zip(axs, ylabels, ys):
    if ax == axs[-1]:
        ax.set_xlabel('Corpus Location', fontsize=AX_FONTSIZE, labelpad=-10)
        ax.set_xticks([0, len(y)])
        # ax.set_xticklabels(['0', '5M'])  # TODO incorrect
        plt.setp(ax.get_xticklabels(), fontsize=AX_FONTSIZE)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel(ylabel, fontsize=LEG_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    plt.setp(ax.get_yticklabels(), fontsize=LEG_FONTSIZE)
    # plot
    ax.plot(y, linewidth=LW, label=ylabel, c='black')
# show
plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
plt.tight_layout(h_pad=HPAD, w_pad=WPAD, pad=PAD)
plt.show()
