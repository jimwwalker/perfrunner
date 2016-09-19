from cbagent.collectors import Collector
from cbagent.collectors.libstats.psstats import PSStats


class PS(Collector):

    COLLECTOR = "atop"  # Legacy / compatibility

    KNOWN_PROCESSES = ("beam.smp", "memcached", "indexer", "projector",
                       "cbq-engine", "goxdcr")

    def __init__(self, settings):
        super(PS, self).__init__(settings)
        self.nodes = settings.hostnames or list(self.get_nodes())

        if hasattr(settings, "fts_server") and settings.fts_server:
            self.KNOWN_PROCESSES = ("beam.smp", "memcached", "cbft",)

        self.ps = PSStats(hosts=self.nodes,
                          user=self.ssh_username, password=self.ssh_password)

    def update_metadata(self):
        self.mc.add_cluster()
        for node in self.nodes:
            self.mc.add_server(node)

    def sample(self):
        for process in self.KNOWN_PROCESSES:
            for node, stats in self.ps.get_samples(process).items():
                if stats:
                    self.update_metric_metadata(stats.keys(), server=node)
                    self.store.append(stats,
                                      cluster=self.cluster, server=node,
                                      collector=self.COLLECTOR)
