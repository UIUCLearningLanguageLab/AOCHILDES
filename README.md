# CHILDESHub

Research code for preparing and analyzing text corpora consisting of child-directed speech.

## Usage

TODO

## Included corpora

I primarily use the two corpora included in this package, `childes-20171212.txt` and `childes-20180319.txt`
They differ in that `childes-20171212.txt` was generated with a few additional processing steps:
1) all title-cased strings were replaced with a single symbol ("TITLED")
2) all words tagged by the Python package `spacy` as referring to a person or organization were replaced by a single symbol
 ("NAME_B"if the word is the first in a span of words referring to a person or organization,
  and "NAME_I" if it is not the first word in a span of words referring to a person or organization)

## Compatibility

Developed on Ubuntu 16.04 and Python 3. 