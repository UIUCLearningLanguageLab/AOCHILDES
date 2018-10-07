from hub import Hub

DPI = 192
HUB_MODE = 'sem'


hub = Hub(mode=HUB_MODE)

probe_context_set_d = {probe: set() for probe in hub.probe_store.types}
for probe in hub.probe_store.types:
    context_set = set(hub.probe_context_terms_dict[probe])
    probe_context_set_d[probe] = context_set

probe_num_overlap_d = {probe: 0 for probe in hub.probe_store.types}
for probe, set_a in probe_context_set_d.items():
    num_overlap = 0
    for set_b in probe_context_set_d.values():
        num_overlap += len(set_a & set_b)
    probe_num_overlap_d[probe] = num_overlap / len(hub.probe_context_terms_dict[probe])
    print(probe, num_overlap)

sorted_by_overlap = sorted(hub.probe_store.types, key=probe_num_overlap_d.get)
print(sorted_by_overlap)
