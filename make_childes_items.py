
from childeshub.params import HubParams, ItemParams
from childeshub.transcripts import Transcripts, PostProcessor
from childeshub.probestore import ProbeStore

probe_store = ProbeStore('sem', HubParams().probes_name)  # exclude probes from normalization


params = ItemParams()
params.lowercase = True

transcripts = Transcripts(params)
proc = PostProcessor(params)

excluded_words = probe_store.types  # exclude probes from processing
proc.to_file(*proc.process(transcripts.age_ordered, excluded_words))



