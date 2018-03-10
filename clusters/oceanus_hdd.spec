[clusters]
oceanus =
    172.23.96.5:kv
    172.23.96.7:kv
    172.23.96.8:cbas
    172.23.96.9:cbas
    172.23.96.57:cbas
    172.23.96.205:cbas

[clients]
hosts =
    172.23.96.22
credentials = root:couchbase

[storage]
data = /data
analytics = /data3 /data4 /data5 /data6 /data7

[credentials]
rest = Administrator:password
ssh = root:couchbase

[parameters]
OS = CentOS 7
CPU = E5-2680 v3 (24 cores)
Memory = 32 GB
Disk = 5 x Seagate HDD
