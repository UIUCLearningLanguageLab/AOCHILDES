"""
Find frequent collocations, that start or end with TARGET word
"""

import math
from collections import Counter

from aochildes.params import AOChildesParams
from aochildes.pipeline import Pipeline

TARGET = 'dr'
LEFT = 0  # distance to left of target
RIGHT = 1  # distance to right of target
N = 30

# get words
params = AOChildesParams()
pipeline = Pipeline(params)
words = [w.lower() for t in pipeline.load_age_ordered_transcripts() for w in t.text.split()]


def update(span, d):
	for w in span:
		if w not in d:
			d[w] = 1
		else:
			d[w] += 1


w2f = Counter(words)
num_words = len(words)

assert TARGET in w2f

col2freq = {}  # empty dictionary for storing collocation frequencies
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
			update(left_span, col2freq)

		# words to the right
		if RIGHT > 0:
			right_span = words[i + 1:end]
			update(right_span, r_freq)
			update(right_span, col2freq)

# compute mutual-info for each collocation
col2mi = {}
for col in col2freq:
	joint_prob = col2freq[col] / num_words
	marginal_prob = (w2f[TARGET] * w2f[col]) / num_words
	mi_score = joint_prob * math.log2(joint_prob / marginal_prob)
	col2mi[col] = mi_score

print(f'total={len(col2mi)}')
print(f'shown={min(len(col2mi), N)}')
for c, mi in sorted(col2mi.items(), key=lambda i: i[1], reverse=True)[:N]:
	print(f'{c:<24} {mi:.6f}')
