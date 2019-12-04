import math
from collections import Counter

from childes.params import Params
from childes.transcripts import Transcripts

TARGET = 'mrs'
LEFT = 0  # distance to left of target
RIGHT = 1  # distance to right of target
N = 30

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
collocate2mi = {}
for collocate in collocate_freq:
	joint_prob = collocate_freq[collocate] / num_words
	marginal_prob = (w2f[TARGET] * w2f[collocate]) / num_words
	mi_score = joint_prob * math.log2(joint_prob / marginal_prob)
	collocate2mi[collocate] = mi_score

print(f'total={len(collocate2mi)}')
print(f'shown={min(len(collocate2mi), N)}')
for c, mi in sorted(collocate2mi.items(), key=lambda i: i[1], reverse=True)[:N]:
	print(f'{c:<24} {mi:.6f}')
