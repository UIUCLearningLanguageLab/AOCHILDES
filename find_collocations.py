import math
from collections import Counter

from childes.params import Params
from childes.transcripts import Transcripts

TARGET = 'thank'
LEFT = 0  # distance to left of target
RIGHT = 1  # distance to right of target
	

# get words
params = Params()
transcripts = Transcripts(params)
texts = transcripts.age_ordered
words = [w.lower() for t in texts for w in t.split()]


def update(span, d):
	for w in span:
		if w not in d:
			d[w] = 1
		else:
			d[w] += 1


w2f = Counter(words)
num_words = len(words)

assert TARGET in w2f

collocate_freq = {}  # empty dictionary for storing collocation frequencies
r_freq = {}  # for hits to the right
l_freq = {}  # for hits to the left
collocate2mi = {}  # for storing the values for whichever stat was used

for i, word in enumerate(words):
	if word == TARGET:
		start = i - LEFT  # beginning of span
		end = i + RIGHT + 1  # end of span
		if start < 0:  # if the left span goes beyond the text
			start = 0  # start at the first word

		# words to the left
		if LEFT > 0:
			left_span = words[start:i]
			update(left_span, l_freq)
			update(left_span, collocate_freq)

		# words to the right
		if RIGHT > 0:
			right_span = words[i + 1:end]
			update(right_span, r_freq)
			update(right_span, collocate_freq)

# compute mutual-info for each collocation
for collocate in collocate_freq:
	observed = collocate_freq[collocate]
	expected = (w2f[TARGET] * w2f[collocate]) / num_words
	mi_score = math.log2(observed / expected)
	collocate2mi[collocate] = mi_score

for c, mi in sorted(collocate2mi.items(), key=lambda i: i[1], reverse=True)[:30]:
	print(f'{c:<24}, {mi:.2f}')
