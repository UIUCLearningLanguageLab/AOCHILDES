from pathlib import Path
from sortedcontainers import SortedSet


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    original = src / 'original_transcripts'
    corpora = root / 'corpora'


class Symbols:
    NAME = '[NAME]'
    PLACE = '[PLACE]'


NAME_COLLOCATIONS = [
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
    ('raggedy', 'andy'),
    ('old', 'macdonald'),
    ('old', 'mcdonald'),
    ('kitty', 'cat'),
    ('richard', 'scarry'),
    ('darth', 'vader'),
    ('anakin', 'skywalker'),
    ('luke', 'skywalker'),
    ('mary', 'beth'),
    ('mary', 'ann'),
    ('mr', 'potato_head'),
    ('humpty', 'dumpty'),
    ('adam', 'smith'),
    ('aunt', 'ada'),
    ('amelia', 'bedelia'),
    ('mr', 'rogers'),
    ('mister', 'rogers'),
    ('mrs', 'rogers'),
    ('miss', 'rogers'),
    ('topham', 'hatt'),
]

SONG = {'doo', 'dee', 'deedee', 'dadada', 'da', 'la_dee_da_dee_dum',
        'ee_yay_ee_yay_oh', 'oompapa'}


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
          'frannie', 'evan', 'aaron', 'nicholas', 'marissa', 'lil', 'awoh', 'hunhunh', 'ruby',
          'sue', 'ann_marie', 'didldow', 'yaya', 'hunm', 'mike', 'uncle_bwww', 'mrs_gwww',
          'stef', 'scott', 'rory', 'josh', 'jo_ann', 'eddy', 'anthony', 'robert',
          'olie', 'nick', 'lucille', 'lindsay', 'greg', 'timmy', 'pierre', 'jeff', 'alexander',
          'abbie', 'julius', 'd_w', 'louise', 'gilly', 'cassandra', 'agra', 'todd', 'timothy',
          'moby', 'malik', 'joseph_p', 'kimberly', 'jenko', 'ericka', 'amelia', 'allison',
          'nora', 'carey', 'blanche',  'uncle_ryan', 'tegama', 'tanta', 'paula', 'panacka',
          'lewis', 'kenny', 'keith', 'jackie', 'gabi', 'diane', 'babykins', 'asha', 'rolf',
          'petey', 'mormor', 'izzy', 'ines', 'ellen', 'edward', 'drew', 'dayja', 'celia',
          'buster', 'betta', 'amye', 'zoey', 'veena', 'pushpa', 'samantha', 'margaret', 'cassie'}

FAMILY = {'mom', 'dad', 'grand_mom', 'grand_dad', 'aunt', 'uncle'}

CHARACTERS = {'dumbo', 'ariel', 'elmo', 'bert', 'ernie', 'santa', 'santa_claus',
              'robin', 'mickey', 'donald', 'cinderella', 'grover', 'maisy', 'snoopy',
              'miffy', 'clifford', 'cookie_monster', 'tom', 'jerry', 'frog', 'toad',
              'hulk', 'paddington', 'harry', 'harry_potter', 'simba', 'piglet', 'leo', 'goofy',
              'tigger', 'minnie', 'dora', 'mcdonald', 'nemo', 'nursie_nellie',
              'froggie', 'eeyore', 'horton', 'alice', 'jay jay', 'snuffy', 'spiderman',
              'huckle', 'pinocchio', 'barney', 'fred', 'dorothy', 'wilbur', 'bambi', 'jesus',
              'batman', 'mickey_mouse', 'chester', 'mufasa', 'goldilocks', 'humpty_dumpty',
              'itsy_bitsy', 'simple_simon', 'popeye', 'goldie', 'dipsy', 'tunky_winky',
              'lincoln', 'lassie', 'old macdonald', 'sully', 'sulley', 'raggedy_ann', 'raggedy_andy', 'kermit',
              'grouchy', 'zach', 'zeek', 'willy_wonka', 'pebbles', 'bam_bam', 'raffi', 'donald_duck',
              'easter_bunny', 'scuffy', 'daffy', 'daffy_duck', 'big_bird', 'dingo', 'fixit',
              'gordon', 'percy', 'care_bear', 'superman', 'e_t', 'yoda', 'blitzen', 'rudolph',
              'he_man', 'vik', 'chewbacca', 'elmer', 'coco', 'alouette', 'winnie_the_pooh', 'piglet',
              'otto', 'noah', 'nala', 'hendrika', 'gaspard', 'lisa', 'darth_vader', 'charlotte',
              'mr_potato_head', 'snow_white', 'rafiki', 'lola', 'benjamin', 'shamu', 'aurora',
              'amelia_bedelia', 'bloat', 'r_two_d_two', 'rapunzel', 'skywalker', 'flintstones'}

names_set = SortedSet()
names_set.update(CHARACTERS)
names_set.update(PEOPLE)
names_set.update(FAMILY)

PLACES = {'vermont', 'mississippi', 'mexico', 'albuquerque', 'france', 'alaska',
          'disneyland', 'cambridge', 'shenandoah', 'pennsylvania', 'ohio', 'pittsburgh',
          'brussels', 'vermont', 'chicago', 'albuquerque', 'alaska', 'england',
          'colorado', 'buffalo', 'michigan', 'providence', 'oregon', 'germany',
          'africa', 'wisconsin', 'tucson'}


PLACE_COLLOCATIONS = [
    ('new', 'york'),
    ('new', 'mexico'),
    ('new', 'hampshire'),
    ('san', 'francisco'),
    ('rhode', 'island'),
    ('king', 'soopers'),
    ('south', 'carolina'),
    ('north', 'carolina'),
    ('moe', '\'s', 'bagels'),
]

places_set = SortedSet()
places_set.update(PLACES)