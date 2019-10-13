# CHILDESHub

Research code for preparing and analyzing text corpora consisting of child-directed speech.

## Usage

TODO

## Included corpus

I primarily use `items/childes-20180319_terms.txt` in my research. 
It was created using only a modest amount of post-processing to preserve as accurately as possible the structure that children actually experience. 
Have a look at `items/childes-20180319_params.yaml` for the parameters used to create the corpus.

* words were lower-cased
* contractions were split
* punctuation was preserved (declaratives, imperatives, and questions)

## Compatibility

Developed on Ubuntu 16.04 and Python 3. 