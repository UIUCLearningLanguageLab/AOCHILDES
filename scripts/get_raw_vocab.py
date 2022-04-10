from collections import Counter

from aochildes.dataset import AOChildesDataSet

data = AOChildesDataSet()

words = []
for _, row in data.pipeline.df.iterrows():
    sentence = row['gloss']
    for word in sentence.split():
        words.append(word)

c = Counter(words)

lines = [f'{w:<36} {f:>9}' for w, f in sorted(c.items(), key=lambda i: i[1], reverse=True)]
print('\n'.join(lines), file=open('raw_vocab.txt', 'w'))
