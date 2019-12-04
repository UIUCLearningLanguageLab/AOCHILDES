from spacy.matcher import PhraseMatcher
from spacy.tokens import Token
from spacy.util import filter_spans

from childes.persons import REAL, FICTIONAL, FAMILY, NICKNAMES
from childes.places import STATES, CITIES, COUNTRIES
from childes.misc import PRODUCTS, SONG

# case-sensitive
persons = set()
persons.update([w.title() for w in REAL])
persons.update([w.title() for w in FICTIONAL])
persons.update([w.title() for w in NICKNAMES])
persons.update([w for w in FAMILY] + [w.title() for w in FAMILY])


class PersonMerger(object):
    def __init__(self, nlp):
        Token.set_extension("is_person", default=False)
        self.matcher = PhraseMatcher(nlp.vocab)
        patterns = [nlp.make_doc(t) for t in persons]
        self.matcher.add("IS_PERSON", None, *patterns)

    def __call__(self, doc):
        # This method is invoked when the component is called on a Doc
        matches = self.matcher(doc)
        spans = []  # Collect the matched spans here
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        with doc.retokenize() as retokenizer:
            spans = filter_spans(spans)  # use only longest non-overlapping spans
            for span in spans:
                retokenizer.merge(span)
                for token in span:
                    token._.is_person = True
        return doc


places = set()
places.update([w.title() for w in STATES])
places.update([w.title() for w in CITIES])
places.update([w.title() for w in COUNTRIES])


class PlacesMerger(object):
    def __init__(self, nlp):
        Token.set_extension("is_place", default=False)
        self.matcher = PhraseMatcher(nlp.vocab)
        patterns = [nlp.make_doc(t) for t in places]
        self.matcher.add("IS_PLACE", None, *patterns)

    def __call__(self, doc):
        # This method is invoked when the component is called on a Doc
        matches = self.matcher(doc)
        spans = []  # Collect the matched spans here
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        with doc.retokenize() as retokenizer:
            spans = filter_spans(spans)  # use only longest non-overlapping spans
            for span in spans:
                retokenizer.merge(span)
                for token in span:
                    token._.is_place = True
        return doc


misc = set()
misc.update([w.title() for w in PRODUCTS])
misc.update([w.title() for w in SONG])


class MiscMerger(object):
    def __init__(self, nlp):
        Token.set_extension("is_misc", default=False)
        self.matcher = PhraseMatcher(nlp.vocab)
        patterns = [nlp.make_doc(t) for t in misc]
        self.matcher.add("IS_MISC", None, *patterns)

    def __call__(self, doc):
        # This method is invoked when the component is called on a Doc
        matches = self.matcher(doc)
        spans = []  # Collect the matched spans here
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        with doc.retokenize() as retokenizer:
            spans = filter_spans(spans)  # use only longest non-overlapping spans
            for span in spans:
                retokenizer.merge(span)
                for token in span:
                    token._.is_misc = True
        return doc