
from childes.params import Params
from childes.transcripts import Transcripts, PostProcessor


params = Params()

transcripts = Transcripts(params)
proc = PostProcessor(params)

proc.to_file(proc.process(transcripts.age_ordered), transcripts.ages)



