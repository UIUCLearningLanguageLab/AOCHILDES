# Create-CHILDES-Corpus

A Python API for retrieving text data consisting of child-directed speech.

Importantly, age of the target child is preserved in ordering of text data.

## Usage

### Raw transcripts


```python
from childes.transcripts import Transcripts

transcripts = Transcripts(sex='m')
```

### Processed transcripts ready for model training

```python
from childes.dataset import ChildesDataSet

dataset = ChildesDataSet()
train_docs, test_docs = dataset.load_docs(num_test_docs=10)
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
