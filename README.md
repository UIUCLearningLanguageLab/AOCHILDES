# AO-CHILDES

Python API for retrieving American-English child-directed speech transcripts,
 ordered by the age of the target child.

## Usage

### Processed transcripts, ordered by age of target child

```python
from aochildes.dataset import AOChildesDataSet

transcripts = AOChildesDataSet().load_transcripts()
```

### Filter male vs. female


```python
from aochildes.dataset import AOChildesDataSet

transcripts = AOChildesDataSet(sex='male').load_transcripts()  # excludes many transcripts not annotated with sex
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

## Compatibility

Developed on Ubuntu 18.04 and Python 3.7. 
