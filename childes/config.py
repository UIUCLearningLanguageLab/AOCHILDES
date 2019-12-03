from pathlib import Path
from sortedcontainers import SortedSet


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    original = src / 'original_transcripts'
    corpora = root / 'corpora'


class Symbols:
    NAME = '[NAME]'
    TITLED = '[TITLED]'


COLLOCATIONS = [
    ('winnie', 'pooh'),
    ('santa', 'claus'),
    ('nursie', 'nellie'),
    ('tunky', 'winky'),
    ('itsy', 'bitsy'),
    ('willy', 'wonka'),
    ('daffy', 'duck'),
    ('easter', 'bunny'),
    ('harry', 'potter'),
    ('raggedy', 'ann'),
    ('old', 'macdonald'),
    ('old', 'mcdonald'),
]


PEOPLE = {'laura', 'mama', 'sarah', 'william', 'ross', 'adam',
          'nomi', 'naima', 'abe', 'alex', 'lily', 'peter', 'nina', 'ethan',
          'nathaniel', 'eve', 'marky', 'mark', 'violet', 'max',
          'paul', 'sally', 'thomas', 'manuela', 'michael',
          'ursula', 'matthew', 'melissa', 'kalie', 'pete', 'bob',
          'jenny', 'eric', 'nana', 'david', 'matty', 'travis', 'fraser', 'megan',
          'sam', 'rachel', 'amanda', 'john', 'jack', 'pat', 'george',
          'olivia', 'erin', 'ryan', 'judy', 'lois', 'trevor', 'joseph',
          'jennifer', 'mary', 'franklin', 'emily', 'phoebe', 'katie', 'cromer',
          'charlie', 'brittany', 'kent', 'linda', 'emma', 'mike', 'arthur', 'nonna',
          'andrew', 'patrick', 'oscar', 'maggie', 'sophie', 'harold', 'james', 'seth',
          'richard', 'jessica', 'hilda', 'jilly', 'gigi', 'georgie', 'oliver', 'bill',
          'karen', 'chantilly', 'jill', 'jamie', 'danny', 'gerry', 'rosie', 'henry',
          'kate', 'joe', 'lisa', 'aislinn', 'laurie', 'sandy', 'lulu', 'catherine',
          'donna', 'colin', 'spencer', 'joey', 'matt', 'becky', 'anne_marie', 'belle',
          'lucy', 'chris', 'billy', 'jane', 'brian', 'liam', 'gordon',
          'nancy', 'ma', 'polly', 'andy', 'ben', 'steve', 'norman', 'kim',
          'joanna', 'jillian', 'gloria', 'elizabeth', 'betty', 'gabby', 'courtney', 'ann',
          'naomi', 'nat', 'tadi', 'luke', 'debbie', 'kathy', 'frank', 'stefan', 'patty', 'annie',
          'tommy', 'sean', 'daniel', 'fred', 'stella', 'marie', 'graham', 'christopher',
          'bobby', 'j_j', 'becca', 'jenell', 'millisandy', 'toddy', 'lynn', 'vaivy',
          'johnny', 'jimmy', 'bobo', 'jen', 'miriam', 'janet', 'sandra', 'cathy', 'zoe',
          'martin', 'ari', 'barbara', 'anne', 'jerry', 'wilson', 'tamar', 'frederick',
          'toby', 'helen', 'elliot', 'jesse', 'dan', 'tippy', 'mrs_gwww', 'diandra', 'ted',
          'leslie', 'elana', 'sara', 'harvey', 'eddie', 'sammy', 'renee', 'mimi',
          'justin', 'jonathan', 'erica', 'wendy', 'susie', 'nathan', 'nanette', 'ellie',
          'leila', 'joshua', 'douglas', 'alec', 'jim', 'isabelle', 'charles', 'tony',
          'julia', 'zack', 'patricia', 'katy', 'griz', 'dick', 'danielle', 'usher',
          'steven', 'sharon', 'meg', 'marty', 'edna', 'weist', 'susan', 'stephen',
          'stan', 'salley', 'felix', 'carl', 'theo', 'liza', 'kirsten', 'heidi', 'esther',
          'dale', 'dabee', 'terry', 'rossy', 'ronnie', 'lia', 'cindy', 'amy', 'tyler',
          'murphy', 'fredrick', 'brenda', 'bobby', 'aunt_dot', 'willie', 'franny', 'ema',
          'craig', 'cara', 'butch', 'brendon', 'theresa', 'sully', 'nicky', 'michelle',
          'gus', 'aunt_carey', 'wanda', 'donny', 'baxter', 'molly', 'larry', 'ian', 'jean',
          'hank', 'betsy', 'avril', 'ais', 'titus', 'tallulah', 'shaun', 'rex', 'pattycake',
          'mel', 'lou', 'gerald', 'aladar', 'shana', 'maryse', 'loi', 'lilly', 'lee', 'gaby',
          'frannie', 'evan', 'aaron', 'nicholas', 'marissa'}

FAMILY = {'mommy', 'daddy', 'grandma', 'mom', 'dad', 'grandpa', 'papa',
          'mummy', 'momma', 'dada', 'auntie', 'gramma', 'grampa', 'untie',
          'granny', 'grammy', 'mum', 'nonno', 'babaji', 'dadaji',
          'grandaddy', 'granma', 'mamma', 'mumma'}

CHARACTERS = {'dumbo', 'ariel', 'elmo', 'bert', 'winnie', 'winnie_pooh', 'ernie', 'santa', 'santa_claus',
              'robin', 'mickey', 'donald', 'cinderella', 'grover', 'maisy', 'snoopy',
              'miffy', 'clifford', 'cookie_monster', 'tom', 'jerry', 'frog', 'toad',
              'hulk', 'paddington', 'harry', 'harry_potter', 'simba', 'piglet', 'goofy',
              'tigger', 'minnie', 'dora', 'mcdonald', 'nemo', 'nursie_nellie',
              'froggie', 'eeyore', 'horton', 'alice', 'jay jay', 'snuffy', 'spiderman',
              'huckle', 'pinocchio', 'barney', 'fred', 'dorothy', 'wilbur', 'bambi', 'jesus',
              'batman', 'mickey_mouse', 'chester', 'mufasa', 'goldilocks', 'humpty_dumpty',
              'itsy_bitsy', 'simple_simon', 'popeye', 'goldie', 'dipsy', 'tunky_winky',
              'lincoln', 'lassie', 'old macdonald', 'sully', 'sulley', 'raggedy_ann', 'kermit',
              'grouchy', 'zach', 'zeek', 'willy_wonka', 'pebbles', 'bam_bam', 'raffi', 'donald_duck',
              'easter_bunny', 'scuffy', 'daffy', 'daffy_duck'}

names_set = SortedSet()
names_set.update(CHARACTERS)
names_set.update(PEOPLE)
names_set.update(FAMILY)