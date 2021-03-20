
from aochildes.persons import REAL, FICTIONAL, FAMILY, NICKNAMES, PETS
from aochildes.places import STATES, CITIES, COUNTRIES
from aochildes.misc import PRODUCTS, SONG, BOOKS_AND_MOVIES

# TODO make PAI for retrieving entities

# case-sensitive
persons = set()
persons.update([w.title() for w in REAL])  # TODO remove .title() and hand-title all words
persons.update([w.title() for w in FICTIONAL])
persons.update([w.title() for w in NICKNAMES])
persons.update([w.title() for w in PETS])
persons.update([w for w in FAMILY] + [w.title() for w in FAMILY])




places = set()
places.update([w.title() for w in STATES])  # TODO remove .title() and hand-title all words
places.update([w.title() for w in CITIES])
places.update([w.title() for w in COUNTRIES])




misc = set()
misc.update([w.title() for w in PRODUCTS])  # TODO remove .title() and hand-title all words
misc.update([w.title() for w in SONG])
misc.update([w.title() for w in BOOKS_AND_MOVIES])
