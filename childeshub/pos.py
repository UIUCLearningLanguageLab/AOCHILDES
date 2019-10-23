from cached_property import cached_property
import string

from childeshub import config

pos2tags = {'verb': ['BES', 'HVS', 'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'noun': ['NN', 'NNS', 'WP'],
            'adverb': ['EX', 'RB', 'RBR', 'RBS', 'WRB'],
            'pronoun': ['PRP'],
            'preposition': ['IN'],
            'conjunction': ['CC'],
            'interjection': ['UH'],
            'determiner': ['DT'],
            'particle': ['POS', 'RP', 'TO'],
            'punctuation': [',', ':', '.', "''", 'HYPH', 'LS', 'NFP'],
            'adjective': ['AFX', 'JJ', 'JJR', 'JJS', 'PDT', 'PRP$', 'WDT', 'WP$'],
            'special': []}


class PartsOfSpeech:

    @cached_property
    def term_tags_dict(self):
        assert len(self.tokens_no_oov) == len(self.tags_no_oov)
        tag_set = set(self.tags_no_oov)
        result = {term: {tag: 0 for tag in tag_set}
                  for term in self.types}
        for term, tag in zip(self.tokens, self.tags_no_oov):
            result[term][tag] += 1
        return result

    @cached_property
    def nouns(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in pos2tags['noun'] \
                    and term not in config.Symbols.all + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def adjectives(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in pos2tags['adjective'] \
                    and term not in config.Symbols.all + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def verbs(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in pos2tags['verb'] \
                    and term not in config.Symbols.all + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def adverbs(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in pos2tags['adverb'] \
                    and term not in config.Symbols.all + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def pronouns(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in pos2tags['pronoun'] \
                    and term not in config.Symbols.all + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def prepositions(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in pos2tags['preposition'] \
                    and term not in config.Symbols.all + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def conjunctions(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in pos2tags['conjunction'] \
                    and term not in config.Symbols.all + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def interjections(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in pos2tags['interjection'] \
                    and term not in config.Symbols.all + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def determiners(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in pos2tags['determiner'] \
                    and term not in config.Symbols.all + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def particles(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in pos2tags['particle'] \
                    and term not in config.Symbols.all + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def punctuations(self):
        result = []
        for term, tags_d in self.train_terms.term_tags_dict.items():
            tag = sorted(tags_d.items(), key=lambda i: i[1])[-1][0]
            if tag in pos2tags['punctuation'] \
                    and term not in config.Symbols.all + list(string.ascii_letters):
                result.append(term)
        return result

    @cached_property
    def specials(self):
        result = [symbol for symbol in config.Symbols.all
                  if symbol in self.train_terms.types]
        return result