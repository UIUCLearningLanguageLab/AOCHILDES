from cached_property import cached_property
from sortedcontainers import SortedSet

from configs import GlobalConfigs


class ProbeStore(object):
    """
    Stores probe-related data.
    """

    def __init__(self, configs_dict, hub_mode, term_id_dict=None):
        print('Initializing probe_store with mode "{}"'.format(hub_mode))
        self.probes_name = configs_dict['{}_probes_name'.format(hub_mode)]
        self.term_id_dict = term_id_dict

    @cached_property
    def probe_cat_dict(self):
        probe_cat_dict = {}
        path = GlobalConfigs.SRC_DIR / 'rnnlab' / 'probes' / '{}.txt'.format(self.probes_name)
        with path.open('r') as f:
            for line in f:
                data = line.strip().strip('\n').split()
                cat = data[0]
                probe = data[1]
                if self.term_id_dict is not None:
                    if probe not in self.term_id_dict:
                        if GlobalConfigs.VERBOSE_PROBE_STORE:
                            print('Probe "{}" not in vocabulary -> Excluded from analysis'.format(probe))
                    else:
                        probe_cat_dict[probe] = cat
                else:
                    probe_cat_dict[probe] = cat
        return probe_cat_dict

    @cached_property
    def types(self):
        probes = sorted(self.probe_cat_dict.keys())
        probe_set = SortedSet(probes)
        print('Num probes: {}'.format(len(probe_set)))
        return probe_set

    @cached_property
    def probe_id_dict(self):
        probe_id_dict = {probe: n for n, probe in enumerate(self.types)}
        return probe_id_dict

    @cached_property
    def cats(self):
        cats = sorted(self.probe_cat_dict.values())
        cat_set = SortedSet(cats)
        return cat_set

    @cached_property
    def cat_id_dict(self):
        cat_id_dict = {cat: n for n, cat in enumerate(self.cats)}
        return cat_id_dict

    @cached_property
    def cat_probe_list_dict(self):
        cat_probe_list_dict = {cat: [probe for probe in self.types if self.probe_cat_dict[probe] == cat]
                               for cat in self.cats}
        return cat_probe_list_dict

    @cached_property
    def num_probes(self):
        num_probes = len(self.types)
        return num_probes

    @cached_property
    def num_cats(self):
        num_cats = len(self.cats)
        return num_cats
