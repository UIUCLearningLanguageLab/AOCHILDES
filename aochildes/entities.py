
from aochildes.persons import REAL, FICTIONAL, FAMILY, NICKNAMES, PETS
from aochildes.places import STATES, CITIES, COUNTRIES
from aochildes.misc import PRODUCTS, SONG, BOOKS_AND_MOVIES

# TODO make API for retrieving entities

# case-sensitive
persons = set()
persons.update([w for w in REAL])
persons.update([w for w in FICTIONAL])
persons.update([w for w in NICKNAMES])
persons.update([w for w in PETS])
persons.update([w for w in FAMILY])

places = set()
places.update([w for w in STATES])
places.update([w for w in CITIES])
places.update([w for w in COUNTRIES])

misc = set()
misc.update([w for w in PRODUCTS])
misc.update([w for w in SONG])
misc.update([w for w in BOOKS_AND_MOVIES])
