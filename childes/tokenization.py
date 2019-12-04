from spacy.symbols import ORTH

special_cases = [
	('J_J.', [{ORTH: "J_J"}, {ORTH: "."}]),
	('J_Js', [{ORTH: "J_J"}, {ORTH: "'s"}]),
	('Joseph_P.', [{ORTH: "Joseph_P"}, {ORTH: "."}]),
	('valentine\'s day', [{ORTH: 'Valentines_Day'}]),
	('mommy\'ll', [{ORTH: 'mommy'}, {ORTH: 'will'}]),
	('Mommy\'ll', [{ORTH: 'Mommy'}, {ORTH: 'will'}]),
	('daddy\'ll', [{ORTH: 'daddy'}, {ORTH: 'will'}]),
	('Daddy\'ll', [{ORTH: 'Daddy'}, {ORTH: 'will'}]),
	('this\'ll',  [{ORTH: 'this'}, {ORTH: 'will'}]),
	('This\'ll',  [{ORTH: 'This'}, {ORTH: 'will'}]),
	('cann\'t',  [{ORTH: 'can'}, {ORTH: 'not'}]),
]


