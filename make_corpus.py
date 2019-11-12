
from childes.params import ItemParams
from childes.transcripts import Transcripts, PostProcessor


params = ItemParams()
params.lowercase = True

transcripts = Transcripts(params)
proc = PostProcessor(params)

excluded_words = []  # TODO exclude probes from processing
proc.to_file(*proc.process(transcripts.age_ordered, excluded_words), transcripts.ages)



