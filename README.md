# AO-CHILDES

Python API for retrieving American-English child-directed speech transcripts,
 ordered by the age of the target child.

## Usage

### Raw transcripts


```python
from aochildes.transcripts import Transcripts

transcripts = Transcripts(sex='m')
```

### Processed transcripts ready for model training

```python
from aochildes.dataset import ChildesDataSet

dataset = ChildesDataSet()
train_docs, test_docs = dataset.load_docs(num_test_docs=10)
```

### List entities

Retrieve sets of entities, like fictional characters mentioned during child-language interactions (e.g. book reading):

```python
from aochildes.persons import FICTIONAL

print(FICTIONAL)
```

## Parameters

A variety of parameters can be set, to influence much processing should be performed on the raw transcripts.
These parameters can be found in `params.py` and should be edited there, directly.
For example, one can set a parameter determining whether or not all utterances with the unicode symbol '�', 'xxx', and 'yyy' are discarded.

## Legacy corpora

The file `legacy/childes-20180319_transcripts.txt` was used by Philip Huebner in his research on training RNNs with age-ordered language input.
It was created using only a modest amount of post-processing to preserve as accurately as possible the structure that children actually experience. 
Have a look at `childes-20180319_params.yaml` for the parameters used to create the corpus.

## Compatibility

Developed on Ubuntu 16.04 and Python 3. 
