"""Create pathologically bad workloads for malloc fragmentation, this case
drives fragmentation in the key/metadata portion of KV-engine.

Rationale
=========

Create a set of documents where the key grows in a way that attempts to ensure
allocations occur over different size-classes. Iterations of the test will then
delete some percentage of the keys whilst storing more keys that are again
sized similarly.

Implementation
==============

A Worker runs in a thread and we spawn a number of these connected by multi
processing queues to form a ring.


            Worker A --> Queue 2 --> Worker B
               ^                        |
               |                        V
     Load -> Queue 1                  Queue 3
               ^                        |
               |                        V
            Worker D <-- Queue 4 <-- Worker C
          (Supervisor)

Also distribute the entire key space amongst the workers, essentially just give
each worker an even slice of the overall key indexes. When the Worker is created
it always sets all of its documents, and when the Worker terminates it does the
same, thus at the start/end of the test we should have exactly num_items *
SIZES keys all with fixed values. During the tests run time the item count
fluctuates as Workers delete and set in the key space.

The test begins when queue 1 is initially populated with n tuples where is the
number of items configured, the tuple has 3 fields defined as:

* First: an item index, which ranges from 0 to items.
* Second: A generator function, see below.
* Third: size used in the last set (begins as 0 and is update as Workers do
  a set and pass on the tuple).

A generator owns a sequence of values to be used in key-size generation.
Calling next() will iterate through the sequence and calling previous() returns
the value of the previous next().

Note that it is the Supervisor worker that populates Queue 1.

Each Worker will pop from the input queue a tuple and perform some work. The
Worker is trying to create keys and generate load in a way that creates severe
fragmentation within KV-engine.

Each Worker generates a key using the size it reads from the generator and the
item index. The key is always formed by taking the item index as a prefix and
generating a string which has a total length of the size. If the index was 13
and the size was 8, the following key is generated:

"13xxxxxx"

Each Worker always calls previous() and next() on the generator.

* If previous() returns a size, generate a key and delete that key.
* If next() returns a size, generate a key and set that key (with a small fixed
size value)

Thus each iteration of a worker deletes the key in its previous size and sets
a the key in its new size.

At the end of the del/set the worker pushes the tuple onto its output queue for
the next worker to operate on.

The Supervisor participates in the del/set as well, but the Supervisor monitors
the tuples it receives.

The Supervisor will stop passing an item on when it calls next() on the received
generator and the call throws StopIteration. Thus the flow around the chain stops
once we have set a key for each index to every size (and d)

After an iteration the supervisor quiesces for a short period (to allow disk
queue to drain and memory to stabilize), then the whole process is repeated for
the given number of iterations.

"""

import multiprocessing
import random
import time
import subprocess

from couchbase import FMT_BYTES, exceptions
from couchbase.connection import Connection
from couchbase.cluster import Cluster
from couchbase.cluster import PasswordAuthenticator

from logger import logger

# SIZE is derived from JEMalloc size-classes, the start size is derived from
# the size of a KV-engine hash-table stored-value which is 56 bytes with no key
# so would have to be allocated from the 64-byte bucket.
# The final size is smaller though to keep our generated keys under the max size
# of memcached keys (250 bytes)

# sizeof(StoredValue) = 56
# 1 byte for collection-id
# 1 byte for key-length which is stored inside the SerialDocKey part of SV
SV_SIZE=56+1+1

SIZES=[]
# 304 isn't a real bin, but we want to hit the 320 bin but not violate the 250
# byte key limit
BINS=[80, 96, 112, 128, 160, 192, 224, 256, 304]

# A small sized and fixed value, ideally values never get allocated into a
# bin we're churning with keys
VALUE_SIZE = 8
VALUE = bytearray(121 for _ in range(VALUE_SIZE))

class KeyFragger:

    def __init__(self, num_items, num_workers, cycles,
                 host, port, bucket, password):

        self.num_items = num_items
        self.num_workers = num_workers

        # Setup the key sizes, go big to small
        for b in reversed(BINS):
            SIZES.append(b - SV_SIZE)

        logger.debug(SIZES)

        # 400 / 4 = 100
        items_per_worker = int(num_items / num_workers)

        # Create queues
        self.queues = list()
        for i in range(self.num_workers):
            self.queues.append(multiprocessing.Queue())

        # Create and spin up workers
        self.workers = list()

        start = 0
        connection = {'host' : host,
                      'port' : port,
                      'bucket' : bucket,
                      'password' : password}

        for i in range(self.num_workers):
            partner = i - 1
            if partner < 0:
                partner = self.num_workers - 1

            if i == self.num_workers - 1:
                # Last one is the Supervisor
                t = Supervisor(number=i,
                               partner=partner,
                               connection=connection,
                               queues=self.queues,
                               in_queue=self.queues[i],
                               out_queue=self.queues[(i + 1) % self.num_workers],
                               num_items=items_per_worker,
                               cycles=cycles)
            else:
                t = Worker(number=i,
                           partner=partner,
                           connection=connection,
                           num_items=items_per_worker,
                           in_queue=self.queues[i],
                           out_queue=self.queues[(i + 1) % self.num_workers],
                           cycles=cycles)
            self.workers.append(t)

    def run(self):
        logger.info('Starting KeyFragger: {} items, {} workers'.format(
            self.num_items, self.num_workers))

        for t in self.workers:
            t.start()
        for t in self.workers:
            t.join()

STOP = 1
GET_READY = 2
SET = 3
DELETE = 4
NEXT_CYCLE = 5

class Worker(multiprocessing.Process):

    def __init__(self,
                 number,
                 partner,
                 connection,
                 num_items,
                 in_queue,
                 out_queue,
                 cycles):
        super().__init__()

        self.id = number
        self.partner = partner
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.connection = connection


        self.num_items = num_items
        self.iteration = 0

        # How many keys per cycle
        self.cycles = cycles
        self.cycle_keys = int(num_items / cycles)

        # each cycle shifts into a new range c1:0-10, c2:11-20 etc...
        self.cycle_offset = 0
        self.state = {}
        # Initialise moves to the next cycle
        self._reset()

        # First cycle offset is 0, undo what _reset did
        self.cycle_offset = 0

    def _reset(self):
        self.cycle_offset += self.cycle_keys
        self.state.clear()
        # We think of each size as a bucket, we will move keys over the buckets
        for s in SIZES:
            # Create a state dictionary for all the 'buckets'
            # Start state is 0.0% full
            # Target state is where we want each bucket to be after all
            # cycles complete, i.e. 25% if we had 4 buckets
            self.state[s] = {'perc' : None,
                              'target' : 100.0/len(SIZES),
                              'curr_items' : 0,
                              'removed' : 0}

        # The 0 or start bucket is seeded, it's 100% for the cycle
        self.state[SIZES[0]]['perc'] = 100.0
        self.state[SIZES[0]]['curr_items'] = self.cycle_keys

    def new_iteration(self):
        return

    def _connect(self):
        """Establish a connection to the Couchbase cluster."""
        cluster = Cluster('http://{}:{}'.format(self.connection['host'], self.connection['port']))
        authenticator = PasswordAuthenticator('Administrator', self.connection['password'])
        cluster.authenticate(authenticator)
        self.client = cluster.open_bucket(self.connection['bucket'])

    def run(self):
        self._connect()

        # Do setup before reading/posting on the queues, so the setup happens
        # 'concurrently' on all worker
        self._setup()

        while True:
            message = self.in_queue.get()

            if message == STOP:
                self.out_queue.put(message)
                break

            if message == NEXT_CYCLE:
                self._reset()
                self.out_queue.put(message)

            if message == GET_READY:
                self.out_queue.put(message)
                continue

            if message == DELETE:
                self.out_queue.put(message)
                self.do_deletes()

            if message == SET:
                self.out_queue.put(message)
                self.do_sets()



    def _set_with_retry(self, key_prefix, keylen):
        success = False
        backoff = 0.01
        # Set key is padded with the Worker's 'A' + id so we get A, B, C, etc...
        key='{prefix:{fill}{align}{width}}'.format(
                            prefix= key_prefix,
                            fill=chr(65+self.id),
                            align='<',
                            width=keylen)
        while not success:
            try:
                self.client.set(key, VALUE, format=FMT_BYTES)
                success = True
            except (exceptions.TimeoutError,
                    exceptions.TemporaryFailError) as e:
                logger.debug('Worker-{0}: Sleeping for {1}s due to {2}'.format(
                    self.start, backoff, e))
                time.sleep(backoff)
                backoff *= 2

    def _del_with_retry(self, key_prefix, keylen):
        success = False
        backoff = 0.01
        # Del key is padded with the Worker's partner ID
        key='{prefix:{fill}{align}{width}}'.format(
                            prefix=key_prefix,
                            fill=chr(65+self.partner),
                            align='<',
                            width=keylen)
        while not success:
            try:
                self.client.remove(key)
                success = True
            except (exceptions.TimeoutError,
                    exceptions.TemporaryFailError) as e:
                logger.debug('Worker-{0}: Sleeping for {1}s due to {2}'.format(
                    self.start, backoff, e))
                time.sleep(backoff)
                backoff *= 2

    def do_deletes(self):
        for s in SIZES[:-1]:
            self._delete(s)

    def do_sets(self):
        for s in range(len(SIZES) - 1):
                self._set(SIZES[s], SIZES[s+1])

    def _setup(self):
        # SETUP. populate the first bucket with 100% of the keys for all cycles
        self._setup_my_slice(SIZES[0], self.cycle_keys * self.cycles)

    # s size of key
    # n number of keys
    def _setup_my_slice(self, s, n):

        logger.debug("Worker: setup issuing {} sets of size:{}.".format(n, s))
        for i in range(n):
            self._set_with_retry(i, s)
        self.state[s]['bound'] = n

    def _delete(self, src):
        if self.state[src]['perc'] and self.state[src]['perc'] <= self.state[src]['target']:
            logger.debug("     Skipping1 DEL from {0} s:{1}".format(src, self.state[src]))
            return
        if self.state[src]['curr_items'] == 0:
            logger.debug("     Skipping2 DEL from {0} s:{1}".format(src, self.state[src]))
            return

        # Compute deletes for this iteration
        # aim to reduce this bucket to target% of cycle keys
        deletes = self.state[src]['curr_items'] - int(self.cycle_keys * (self.state[src]['target']/100.0))

        start = self.cycle_offset + self.state[src]['removed']

        logger.debug("Worker: issuing {} deletes of key size:{} range:{}-{}. state:{}".format(deletes, src, start, start+deletes, self.state[src]))
        bound = self.state[src]['bound']
        for i in range(start, (start+deletes)):
             self._del_with_retry(i, src)

        self.state[src]['curr_items'] -= deletes
        self.state[src]['perc'] = (self.state[src]['curr_items'] / self.cycle_keys) * 100.0
        self.state[src]['removed'] += deletes

    def _set(self, src, dst):
        # source is empty
        if self.state[src]['curr_items'] == 0 or self.state[src]['removed'] == 0:
            logger.debug("     Skipping1 SET from {0} to {1} s:{2} d:{3}".format(src, dst,  self.state[src],  self.state[dst] ))
            return
        # dest is at (or below) target
        if self.state[dst]['perc'] and self.state[dst]['perc'] <= self.state[dst]['target']:
            logger.debug("     Skipping2 SET from {0} to {1} s:{2} d:{3}".format(src, dst,  self.state[src],  self.state[dst] ))
            return

        sets = self.state[src]['removed']

        start = self.cycle_offset

        logger.debug("Worker: issuing {} sets of size:{}. offset:{} range:{}-{} ".format(sets, dst,  self.cycle_offset, start, start+sets))

        # Set into dst what was removed from src
        for i in range(start, start + sets):
             self._set_with_retry(i, dst)


        # Update dst state
        self.state[dst]['perc'] = (sets / self.cycle_keys) * 100.0
        self.state[dst]['curr_items'] += sets
        self.state[dst]['bound'] = start + sets

class Supervisor(Worker):

    SLEEP_TIME = 2

    def __init__(self,
                 number,
                 partner,
                 connection,
                 queues,
                 in_queue,
                 out_queue,
                 num_items,
                 cycles):
        super().__init__(number,
                         partner,
                         connection,
                         num_items,
                         in_queue,
                         out_queue,
                         cycles)
        self.queues = queues
        self.num_items = num_items

        # The Supervisor work loop runs in a SET phase then switches to DEL
        # hence why we double based on the number of SIZES
        self.num_iterations = len(SIZES) * 2
        self.cycles = cycles

    def run(self):
        logger.info('Starting KeyFragger supervisor')

        self._connect()

        self.out_queue.put(GET_READY)

        self._setup()

        sleepTime = 0

        for cycle in range(self.cycles):
            message = self.in_queue.get()
            if message != GET_READY and message != NEXT_CYCLE:
                logger.warn(
                    "Supervisor: cycle {0} expected {1} or {2} but got {3}"
                        .format(cycle, GET_READY, NEXT_CYCLE, message))
            else:
                logger.info("Supervisor: All workers completed cycle setup")


            mode = DELETE
            for iteration in range(self.num_iterations):
                if mode == DELETE:
                    logger.info('Supervisor: Running delete phase')
                    self.out_queue.put(DELETE)
                    self.do_deletes()
                    # Deletes need to persist for the memory to be freed
                    sleepTime = 0

                if mode == SET:
                    logger.info('Supervisor: Running set phase')
                    self.out_queue.put(SET)
                    self.do_sets()
                    sleepTime = 0

                message = self.in_queue.get()

                if message != mode:
                    logger.warn("Supervisor: run expected {0} got {1}".format(mode, message))


                logger.info('Supervisor: Completed iteration {0}/{1}'.format(
                    iteration + 1, self.num_iterations))
                logger.info('Supervisor:  Sleeping for {}s'.format(sleepTime))
                time.sleep(sleepTime)

                # Switch phase
                if mode == DELETE:
                    mode = SET
                else:
                    mode = DELETE

            logger.info("Supervisor Finished cycle {0}/{1}".format(cycle+1, self.cycles))
            self.out_queue.put(NEXT_CYCLE)
            self._reset()

        logger.info('Supervisor: issuing STOP')
        self.out_queue.put(STOP)

if __name__ == '__main__':

    # Small smoketest
    KeyFragger(num_items=100000,
               num_workers=8,
               cycles=1,
               host='localhost',
               port=9000,
               bucket='bucket-1',
               password='asdasd').run()