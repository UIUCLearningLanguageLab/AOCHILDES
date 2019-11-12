# Create-CHILDES-Corpus

Research code for preparing text corpora consisting of child-directed speech.
Each line in the resulting text file is a transcript.
Importantly, transcripts are always ordered by age of the target child.

## Usage

To create a text corpus, execute:

```bash
python3 make_corpus.py
```

To expose API within Python:

```python
from childes.transcripts import Transcripts

transcripts = Transcripts(sex='m')
```

## Included corpora

The file `items/childes-20180319_terms.txt` was used by Philip Huebner in his research on training RNNs with age-ordered language input.
It was created using only a modest amount of post-processing to preserve as accurately as possible the structure that children actually experience. 
Have a look at `items/childes-20180319_params.yaml` for the parameters used to create the corpus.

* words were lower-cased
* contractions were split
* punctuation was preserved (declaratives, imperatives, and questions)

## To-do

* exclude probe words (from CategoryEval) from post-processing

## Compatibility

Developed on Ubuntu 16.04 and Python 3. 
