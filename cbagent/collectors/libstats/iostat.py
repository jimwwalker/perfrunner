from cbagent.collectors.libstats.remotestats import RemoteStats, parallel_task


class IOStat(RemoteStats):

    METRICS = (
        ("rps", "r/s", 1),
        ("wps", "w/s", 1),
        ("rbps", "rkB/s", 1024),  # kB -> B
        ("wbps", "wkB/s", 1024),  # kB -> B
        ("avgqusz", "avgqu-sz", 1),
        ("await", "await", 1),
        ("util", "%util", 1),
    )

    def get_device_name(self, partition):
        for path in (partition, '/'):
            stdout = self.run("df '{}'| head -2 | tail -1".format(path),
                              warn_only=True, quiet=True)
            if not stdout.return_code:
                name = stdout.split()[0]
                if name.startswith('/dev/mapper/'):
                    return name.split('/dev/mapper/')[1]
                else:
                    return name

    def get_iostat(self, device):
        stdout = self.run(
            "iostat -dkxyN 1 1 {} | grep -v '^$' | tail -n 2".format(device)
        )
        stdout = stdout.split()
        header = stdout[:len(stdout) // 2]
        data = dict()
        for i, value in enumerate(stdout[len(stdout) // 2:]):
            data[header[i]] = value
        return data

    @parallel_task(server_side=True)
    def get_server_samples(self, partitions: dict) -> dict:
        return self.get_samples(partitions['server'])

    @parallel_task(server_side=False)
    def get_client_samples(self, partitions: dict) -> dict:
        return self.get_samples(partitions['client'])

    def get_samples(self, partitions: dict) -> dict:
        samples = {}

        for purpose, partition in partitions.items():
            device = self.get_device_name(partition)
            data = self.get_iostat(device)
            for shorthand, metric, multiplier in self.METRICS:
                key = "{}_{}".format(purpose, shorthand)
                samples[key] = float(data[metric]) * multiplier

        return samples


class DiskStats(IOStat):

    def get_disk_stats(self, device: str):
        device_name = device.split('/')[-1]

        # https://www.kernel.org/doc/Documentation/ABI/testing/procfs-diskstats
        stdout = self.run("grep '{}' /proc/diskstats".format(device_name))
        stats = stdout.split()
        sectors_read, sectors_written = int(stats[5]), int(stats[9])

        # https://www.kernel.org/doc/Documentation/block/queue-sysfs.txt
        parent = self.run('lsblk -no pkname {}'.format(device)).strip()
        stdout = self.run('cat /sys/block/{}/queue/hw_sector_size'.format(parent))
        sector_size = int(stdout)

        return sectors_read * sector_size, sectors_written * sector_size

    @parallel_task(server_side=True)
    def get_server_samples(self, partitions: dict) -> dict:
        return self.get_samples(partitions['server'])

    def get_samples(self, partitions: dict) -> dict:
        samples = {}

        for purpose, partition in partitions.items():
            device = self.get_device_name(partition)
            bytes_read, bytes_written = self.get_disk_stats(device)

            samples[purpose + '_bytes_read'] = bytes_read
            samples[purpose + '_bytes_written'] = bytes_written

        return samples
