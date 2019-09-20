import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from childeshub.hub import Hub

HUB_MODE = 'sem'

hub = Hub(mode=HUB_MODE)


def plot(*terms_list, dpi=192, ax_fontsize=12, fit_line=False, labels=('1st half', '2nd half'),
         leg_fontsize=8, is_log=False, title='', num_bins=100, rolling_mean=0):
    # fig
    fig, ax = plt.subplots(dpi=dpi)
    plt.title(title)
    ax.set_xlabel('Corpus Location', fontsize=ax_fontsize)
    ax.set_ylabel('{} Frequency'.format('Log' if is_log else ''), fontsize=ax_fontsize)
    # plot
    linestyles = ['-', ':']
    for terms, ls, label in zip(terms_list, linestyles, labels):
        (x, y) = hub.make_locs_xy(terms, num_bins=num_bins)
        if is_log:
            y = np.log(np.clip(y, 1, np.max(y)))
        if rolling_mean > 0:
            y = hub.roll_mean(y, rolling_mean)
        ax.plot(x, y, label=label, linestyle=ls, color='black')
        # fit line
        if fit_line:
            y_fitted = hub.fit_line(x, y)
            ax.plot(x, y_fitted, linestyle='--')
    plt.legend(loc='upper center', fontsize=leg_fontsize, frameon=False,
               bbox_to_anchor=(0.5, 0.1), ncol=3)
    plt.show()


half_num_types = hub.params.num_types // 2
pos_terms_list = [hub.nouns, hub.verbs, hub.adjectives]
titles = ['Nouns', 'Verbs', 'Adjectives']
num_titles = len(titles)

# location
loc_sorted_terms = sorted(hub.train_terms.types, key=hub.calc_avg_reordered_loc)
plot(loc_sorted_terms[:half_num_types], loc_sorted_terms[-half_num_types:],
     title='Location')

# frequency
freq_sorted_terms = list(zip(*hub.train_terms.term_freq_dict.most_common(
    hub.params.num_types)))[0]
plot(freq_sorted_terms[:half_num_types], title='Most Frequent',
     num_bins=1000, rolling_mean=500)
plot(freq_sorted_terms[-half_num_types:], title='least Frequent',
     num_bins=1000, rolling_mean=500)

# loc_asymmetry
loc_asymmetry_sorted_terms = sorted(hub.train_terms.types, key=hub.calc_loc_asymmetry)
plot(loc_asymmetry_sorted_terms[:half_num_types], loc_asymmetry_sorted_terms[-half_num_types:],
     title='Loc Asymmetry')

################################# figs

# grammatical cat fig
LW = 1
AX_FONTSIZE = 7
LEG_FONTSIZE = 6
FIGSIZE = (3.2, 0.8 * num_titles)
DPI = 192
IS_LOG = False
WSPACE = 0.0
HSPACE = 0.0
WPAD = 0.0
HPAD = 0.0
PAD = 0.2
ROLLING_MEAN = 10
NUM_BINS = 100
_, axs = plt.subplots(num_titles, 1, sharex='all',
                        dpi=DPI, figsize=FIGSIZE)
if num_titles == 1:
    axs = [axs]
for ax, pos_terms, title in zip(axs, pos_terms_list, titles):
    if ax == axs[-1]:
        ax.set_xlabel('Corpus Location', fontsize=AX_FONTSIZE, labelpad=-7)
        ax.set_xticks([1, 5000000])
        ax.set_xticklabels(['0', '5M'])  # TODO incorrect
    else:
        ax.set_xticks([])
    if ax == axs[-num_titles // 2]:
        ax.set_ylabel('Frequency', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    plt.setp(ax.get_yticklabels(), fontsize=LEG_FONTSIZE)
    plt.setp(ax.get_xticklabels(), fontsize=LEG_FONTSIZE)
    ax.set_title(title, fontsize=LEG_FONTSIZE, y=0.75, x=0.75)
    # plot
    ax.axvline(x=hub.midpoint_loc, color='grey', linestyle=':')
    pos_terms_sorted_by_loc = sorted(pos_terms, key=hub.calc_loc_asymmetry)
    half = len(pos_terms_sorted_by_loc) // 2
    colors = ['black', 'grey']
    terms_list = [pos_terms_sorted_by_loc[:half], pos_terms_sorted_by_loc[-half:]]
    labels = ['1st half', '2nd half']
    for terms, c, label in zip(terms_list, colors, labels):
        (x, y) = hub.make_locs_xy(terms, num_bins=NUM_BINS)
        if IS_LOG:
            y = np.log(np.clip(y, 1, np.max(y)))
        if ROLLING_MEAN > 0:
            y = hub.roll_mean(y, ROLLING_MEAN)
        ax.plot(x, y, label=label, color=c, linewidth=LW)
# show
plt.legend(loc='upper center', fontsize=LEG_FONTSIZE, frameon=False,
               bbox_to_anchor=(0.48, 0.3), ncol=3)
plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
plt.tight_layout(h_pad=HPAD, w_pad=WPAD, pad=PAD)
plt.show()

# loc_asymmetry fig
NUM_MOST_FREQ = 100
REMOVE = ['oh']
IS_LOG = False
SUBTRACT_MEAN = True
ROLLING_MEAN = 50
NUM_LINES = 6
LW = 1
loc_asymmetry_sorted_terms_most_freq = [t for t in loc_asymmetry_sorted_terms
                                        if t in freq_sorted_terms[:NUM_MOST_FREQ]]
for r in REMOVE:
    loc_asymmetry_sorted_terms_most_freq.remove(r)
titles = ['smallest slope', 'largest slope']
half = len(loc_asymmetry_sorted_terms_most_freq) // 2
terms_list = [loc_asymmetry_sorted_terms_most_freq[:half],
              loc_asymmetry_sorted_terms_most_freq[-half:],]
_, axs = plt.subplots(2, 1, sharex='all', dpi=DPI, figsize=FIGSIZE)
for ax, title, terms in zip(axs, titles, terms_list):
    if ax == axs[1]:
        ax.set_xlabel('Corpus Location', fontsize=AX_FONTSIZE)
    ax.set_ylabel('Mean-Norm. Freq', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    plt.setp(ax.get_yticklabels(), fontsize=LEG_FONTSIZE)
    plt.setp(ax.get_xticklabels(), fontsize=LEG_FONTSIZE)
    ax.set_title(title, fontsize=LEG_FONTSIZE, y=0.9)
    # plot
    colors = sns.color_palette("hls", NUM_LINES)
    for term, color in zip(terms[:NUM_LINES], colors):
        (x, y) = hub.make_locs_xy([term], num_bins=NUM_BINS)
        if IS_LOG:
            y = np.log(np.clip(y, 1, np.max(y)))
        if SUBTRACT_MEAN:
            y = np.subtract(y, np.mean(y))
        if ROLLING_MEAN > 0:
            y = hub.roll_mean(y, ROLLING_MEAN)
        ax.plot(x, y, label='"{}"'.format(term), c=color, linewidth=LW)
    ax.legend(loc='best', fontsize=LEG_FONTSIZE, frameon=False, ncol=NUM_LINES // 3)
# show
plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
plt.tight_layout(h_pad=HPAD, w_pad=WPAD, pad=PAD)
plt.show()

# single term fig
terms = ['bottle', 'story']
num_terms = len(terms)
LW = 0.5
IS_XTICKLABELS = False
AX_FONTSIZE = 7
LEG_FONTSIZE = 6
FIGSIZE = (3.2, 0.8 * num_terms)
DPI = 192
IS_LOG = False
WSPACE = 0.0
HSPACE = 0.0
WPAD = 0.0
HPAD = 0.0
PAD = 0.2
ROLLING_MEAN = 10
NUM_BINS = 100
_, axs = plt.subplots(num_terms, 1, sharex='all',
                        dpi=DPI, figsize=FIGSIZE)
if num_terms == 1:
    axs = [axs]
for ax, term in zip(axs, terms):
    if ax == axs[-1]:
        ax.set_xlabel('Corpus Location', fontsize=AX_FONTSIZE)
    if ax == axs[-num_terms // 2]:
        ax.set_ylabel('Frequency', fontsize=AX_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    if not IS_XTICKLABELS:
        ax.set_xticks([])
        ax.set_xticklabels([])
    plt.setp(ax.get_yticklabels(), fontsize=LEG_FONTSIZE)
    plt.setp(ax.get_xticklabels(), fontsize=LEG_FONTSIZE)
    ax.set_title(term, fontsize=LEG_FONTSIZE, y=0.9)
    # plot
    ax.axvline(x=hub.midpoint_loc, color='grey', linestyle=':')
    (x, y) = hub.make_locs_xy([term], num_bins=NUM_BINS)
    if IS_LOG:
        y = np.log(np.clip(y, 1, np.max(y)))
    if ROLLING_MEAN > 0:
        y = hub.roll_mean(y, ROLLING_MEAN)
    ax.plot(x, y, color='black', linewidth=LW)
# show
plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
plt.tight_layout(h_pad=HPAD, w_pad=WPAD, pad=PAD)
plt.show()