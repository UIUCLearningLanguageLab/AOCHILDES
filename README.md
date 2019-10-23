# CHILDESHub

Research code for preparing text corpora consisting of child-directed speech.
Each line in the resulting text file is a transcript.
Importantly, transcripts are always ordered by age of the target child.

## Usage

To create a text corpus, execute:

```bash
python3 make_corpus.py
```

## Included corpora

I primarily use `items/childes-20180319_terms.txt` in my research. 
It was created using only a modest amount of post-processing to preserve as accurately as possible the structure that children actually experience. 
Have a look at `items/childes-20180319_params.yaml` for the parameters used to create the corpus.

* words were lower-cased
* contractions were split
* punctuation was preserved (declaratives, imperatives, and questions)

## To-do

* exclude probe words (from CategoryEval from pos-processing)

## Compatibility

Developed on Ubuntu 16.04 and Python 3. 