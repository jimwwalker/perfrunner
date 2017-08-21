from logger import logger
from perfrunner.helpers.cbmonitor import timeit, with_stats
from perfrunner.helpers.worker import (
    pillowfight_data_load_task,
    pillowfight_task,
)
from perfrunner.tests import PerfTest
from perfrunner.workloads.pathoGen import PathoGen
from perfrunner.workloads.tcmalloc import WorkloadGen


class KVTest(PerfTest):

    @with_stats
    def access(self, *args):
        super().sleep()

    def reset_kv_stats(self):
        for server in self.cluster_spec.servers:
            for bucket in self.test_config.buckets:
                port = self.rest.get_memcached_port(server)
                self.memcached.reset_stats(server, port, bucket)

    def run(self):
        self.load()
        self.wait_for_persistence()

        self.hot_load()

        self.reset_kv_stats()

        self.access_bg()
        self.access()

        self.report_kpi()


class ReadLatencyTest(KVTest):

    """Enable reporting of GET latency."""

    COLLECTORS = {'latency': True}

    def _report_kpi(self):
        self.reporter.post(
            *self.metrics.kv_latency(operation='get')
        )


class MixedLatencyTest(ReadLatencyTest):

    """Enable reporting of GET and SET latency."""

    def _report_kpi(self):
        for operation in ('get', 'set'):
            self.reporter.post(
                *self.metrics.kv_latency(operation=operation)
            )


class ReadLatencyDGMTest(ReadLatencyTest):

    COLLECTORS = {'latency': True, 'net': False, 'page_cache': True}

    def _report_kpi(self):
        self.reporter.post(
            *self.metrics.kv_latency(operation='get', percentile=99.9)
        )


class DurabilityTest(KVTest):

    """Enable reporting of persistTo=1 and replicateTo=1 latency."""

    COLLECTORS = {'durability': True}

    def _report_kpi(self):
        for operation in ('replicate_to', 'persist_to'):
            self.reporter.post(
                *self.metrics.kv_latency(operation=operation,
                                         dbname='durability')
            )


class SubDocTest(MixedLatencyTest):

    """Enable reporting of SubDoc latency."""

    COLLECTORS = {'subdoc_latency': True}


class XATTRTest(MixedLatencyTest):

    """Enable reporting of XATTR latency."""

    COLLECTORS = {'xattr_latency': True}

    def add_xattr(self):
        access_settings = self.test_config.access_settings
        access_settings.seq_updates = True

        PerfTest.access(self, settings=access_settings)

    def run(self):
        self.load()
        self.wait_for_persistence()

        self.add_xattr()

        self.access_bg()
        self.access()

        self.report_kpi()


class BgFetcherTest(KVTest):

    """Enable reporting of average BgFetcher wait time (disk fetches)."""

    COLLECTORS = {'net': False, 'page_cache': True}

    def _report_kpi(self):
        self.reporter.post(
            *self.metrics.avg_bg_wait_time()
        )


class DrainTest(KVTest):

    """Enable reporting of average disk write queue size."""

    COLLECTORS = {'net': False, 'page_cache': True}

    def _report_kpi(self):
        self.reporter.post(
            *self.metrics.avg_disk_write_queue()
        )


class InitialLoadTest(DrainTest):

    @with_stats
    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

    def run(self):
        self.load()

        self.report_kpi()


class BeamRssTest(KVTest):

    """Enable reporting of Erlang (beam.smp process) memory usage."""

    def _report_kpi(self):
        self.reporter.post(
            *self.metrics.max_beam_rss()
        )


class WarmupTest(PerfTest):

    """Measure the time it takes to perform cluster warm up."""

    COLLECTORS = {'net': False, 'page_cache': True}

    def access(self, *args, **kwargs):
        super().sleep()

    @with_stats
    def warmup(self):
        self.remote.stop_server()
        self.remote.drop_caches()

        return self._warmup()

    @timeit
    def _warmup(self):
        self.remote.start_server()
        for master in self.cluster_spec.masters:
            for bucket in self.test_config.buckets:
                self.monitor.monitor_warmup(self.memcached, master, bucket)

    def _report_kpi(self, time_elapsed):
        self.reporter.post(
            *self.metrics.elapsed_time(time_elapsed)
        )

    def run(self):
        self.load()
        self.wait_for_persistence()

        self.access_bg()
        self.access()
        self.wait_for_persistence()

        time_elapsed = self.warmup()

        self.report_kpi(time_elapsed)


class FragmentationTest(PerfTest):

    """Implement the append-only workload.

    Scenario:
    1. Single node.
    2. Load X items, 700-1400 bytes, average 1KB (11-22 fields).
    3. Append data
        3.1. Mark first 80% of items as working set.
        3.2. Randomly update 75% of items in working set by adding 1 field at a time (62 bytes).
        3.3. Mark first 40% of items as working set.
        3.4. Randomly update 75% of items in working set by adding 1 field at a time (62 bytes).
        3.5. Mark first 20% of items as working set.
        3.6. Randomly update 75% of items in working set by adding 1 field at a time (62 bytes).
    4. Repeat step #3 5 times.

    See workloads/tcmalloc.py for details.

    Scenario described above allows to spot issues with memory/allocator
    fragmentation.
    """

    COLLECTORS = {'net': False}

    @with_stats
    def load_and_append(self):
        password = self.test_config.bucket.password
        WorkloadGen(self.test_config.load_settings.items,
                    self.master_node, self.test_config.buckets[0],
                    password).run()

    def calc_fragmentation_ratio(self) -> float:
        ratios = []
        for target in self.target_iterator:
            port = self.rest.get_memcached_port(target.node)
            stats = self.memcached.get_stats(target.node, port, target.bucket,
                                             stats='memory')
            ratio = int(stats[b'mem_used']) / int(stats[b'total_heap_bytes'])
            ratios.append(ratio)
        ratio = 100 * (1 - sum(ratios) / len(ratios))
        ratio = round(ratio, 1)
        logger.info('Fragmentation: {}'.format(ratio))
        return ratio

    def _report_kpi(self):
        ratio = self.calc_fragmentation_ratio()

        self.reporter.post(
            *self.metrics.fragmentation_ratio(ratio)
        )

    def run(self):
        self.load_and_append()

        self.report_kpi()


class FragmentationLargeTest(FragmentationTest):

    @with_stats
    def load_and_append(self):
        password = self.test_config.bucket.password
        WorkloadGen(self.test_config.load_settings.items,
                    self.master_node, self.test_config.buckets[0], password,
                    small=False).run()


class MemUsedTest(KVTest):

    ALL_BUCKETS = True

    def reset_kv_stats(self):
        pass

    def _report_kpi(self):
        for metric in ('max', 'min'):
            self.reporter.post(
                *self.metrics.mem_used(metric)
            )
            self.reporter.post(
                *self.metrics.mem_used(metric)
            )


class PathoGenTest(FragmentationTest):

    @with_stats
    def access(self, *args):
        for target in self.target_iterator:
            pg = PathoGen(num_items=self.test_config.load_settings.items,
                          num_workers=self.test_config.load_settings.workers,
                          num_iterations=self.test_config.load_settings.iterations,
                          frozen_mode=False,
                          host=target.node, port=8091,
                          bucket=target.bucket, password=target.password)
            pg.run()

    def _report_kpi(self):
        self.reporter.post(
            *self.metrics.avg_memcached_rss()
        )
        self.reporter.post(
            *self.metrics.max_memcached_rss()
        )

    def run(self):
        self.access()

        self.report_kpi()


class PathoGenFrozenTest(PathoGenTest):

    @with_stats
    def access(self):
        for target in self.target_iterator:
            pg = PathoGen(num_items=self.test_config.load_settings.items,
                          num_workers=self.test_config.load_settings.workers,
                          num_iterations=self.test_config.load_settings.iterations,
                          frozen_mode=True,
                          host=target.node, port=8091,
                          bucket=target.bucket, password=target.password)
            pg.run()


class ThroughputTest(KVTest):

    COLLECTORS = {'latency': True}

    def _measure_curr_ops(self) -> int:
        ops = 0
        for bucket in self.test_config.buckets:
            for server in self.cluster_spec.servers:
                port = self.rest.get_memcached_port(server)

                stats = self.memcached.get_stats(server, port, bucket)
                for stat in b'cmd_get', b'cmd_set':
                    ops += int(stats[stat])
        return ops

    def _report_kpi(self):
        total_ops = self._measure_curr_ops()

        self.reporter.post(
            *self.metrics.kv_throughput(total_ops)
        )


class EvictionTest(KVTest):

    COLLECTORS = {'net': False}

    def reset_kv_stats(self):
        pass

    def _measure_ejected_items(self) -> int:
        ejected_items = 0
        for bucket in self.test_config.buckets:
            for hostname, _ in self.rest.get_node_stats(self.master_node,
                                                        bucket):
                host = hostname.split(':')[0]
                port = self.rest.get_memcached_port(host)

                stats = self.memcached.get_stats(host, port, bucket)

                ejected_items += int(stats[b'vb_active_auto_delete_count'])
                ejected_items += int(stats[b'vb_pending_auto_delete_count'])
                ejected_items += int(stats[b'vb_replica_auto_delete_count'])
        return ejected_items

    def _report_kpi(self):
        ejected_items = self._measure_ejected_items()

        self.reporter.post(
            *self.metrics.kv_throughput(ejected_items)
        )


class PillowFightTest(PerfTest):

    """Use cbc-pillowfight from libcouchbase to drive cluster."""

    ALL_BUCKETS = True

    def load(self, *args):
        PerfTest.load(self, task=pillowfight_data_load_task)

    @with_stats
    def access(self, *args):
        self.download_certificate()

        PerfTest.access(self, task=pillowfight_task)

    def _report_kpi(self):
        self.reporter.post(
            *self.metrics.max_ops()
        )

    def run(self):
        self.load()
        self.wait_for_persistence()

        self.access()

        self.report_kpi()


class CompactionTest(KVTest):

    COLLECTORS = {'net': False}

    @with_stats
    @timeit
    def compact(self):
        self.compact_bucket()

    def _report_kpi(self, time_elapsed):
        self.reporter.post(
            *self.metrics.elapsed_time(time_elapsed)
        )

    def run(self):
        self.load()
        self.wait_for_persistence()

        self.hot_load()

        self.access_bg()

        time_elapsed = self.compact()

        self.report_kpi(time_elapsed)


class MemoryOverheadTest(PillowFightTest):

    COLLECTORS = {'iostat': False, 'net': False}

    PF_KEY_SIZE = 20

    def _report_kpi(self):
        self.reporter.post(
            *self.metrics.memory_overhead(key_size=self.PF_KEY_SIZE)
        )

    @with_stats
    def access(self, *args):
        self.sleep()
