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

from couchbase import FMT_BYTES, Couchbase, exceptions

from logger import logger

# SIZE is derived from JEMalloc size-classes, the start size is derived from
# the size of a KV-engine hash-table stored-value which is 56 bytes with no key
# so would have to be allocated from the 64-byte bucket.
# The final size is smaller though to keep our generated keys under the max size
# of memcached keys (250 bytes)

# The test is geared around some knowledge of KV-engine
# * A stored-value is 56 bytes (as of git-sha 50b6ead0)
# * JEMalloc is the memory allocator
# The SIZES list determines what length keys to create, this is the total padded
# size, i.e. len(prefix + padding) == size.
# The prefix is our current key index, which is just a number and thus ranges
# from 0 to a theoretical max of 9,999,999.
# Hence the first size of 8 allows for keys "0xxxxxxx" and "9999999x" and these
# should in-theory be allocated out of the 64-byte bucket (6 + 56 = 62).
# The last size is chosen to ensure we generate legal memcached keys
SIZES1 = (8,  # Total 62, expect to hit the 64-byte bucket
         16, # Total 72, expect to hit the 80-byte bucket
         32, # Total 88, expect to hit the 96-byte bucket
         64, # Total 120, expect to hit the 128-byte bucket
         80, # Total 136, expect to hit the 160-byte bucket
         128, # Total 184, expect to hit the 192-byte bucket
         160, # Total 216, expect to hit the 224-byte bucket
         192, # Total 248, expect to hit the 256-byte bucket
         224) # Total 280, expect to hit the 320-byte bucket

SIZES = ((8, 9, 10, 11, 12, 13, 14, 15),
         (32, 34, 36, 38, 40, 42, 44, 46),
         (80, 83, 86, 89, 92, 95, 98, 101),
    (128, 131, 134, 137, 140,
    143,
    146,
    149),
    (224,
    231,
    238,
    243, # approaching max key
    223,
    232,
    235,
    239))

# A small sized and fixed value, ideally values never get allocated into a
# bucket we're churning with keys
VALUE_SIZE = 8
VALUE = bytearray(121 for _ in range(VALUE_SIZE))

class KeyFragger:

    def __init__(self, num_items, num_workers, num_iterations, frozen_mode,
                 host, port, bucket, password):

        self.num_items = num_items
        self.num_workers = num_workers

        # 5000 items, 20 workers, 6 SIZES
        # 250 per worker
        # 41 per class,
        items_per_worker = int(num_items / num_workers)

        # Create queues
        self.queues = list()
        for i in range(self.num_workers):
            self.queues.append(multiprocessing.Queue())

        # Create and spin up workers
        self.workers = list()

        start = 0
        key_slice_size = int(num_items/num_workers)
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
                               num_iterations=num_iterations,
                               key_slice=(start, num_items))
            else:
                t = Worker(number=i,
                           partner=partner,
                           connection=connection,
                           num_items=items_per_worker,
                           in_queue=self.queues[i],
                           out_queue=self.queues[(i + 1) % self.num_workers],
                           key_slice=(start, key_slice_size*i))
            start = (key_slice_size*i)
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
ITERATION_DONE=5

class Worker(multiprocessing.Process):

    def __init__(self,
                 number,
                 partner,
                 connection,
                 num_items,
                 in_queue,
                 out_queue,
                 key_slice):
        super().__init__()
        self.id = number
        self.partner = partner
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.connection = connection
        self.key_slice = key_slice
        self.num_items = num_items
        self.iteration = 0

    def _connect(self):
        """Establish a connection to the Couchbase cluster."""
        self.client = Couchbase.connect(host=self.connection['host'],
                                        port=self.connection['port'],
                                        bucket=self.connection['bucket'],
                                        password=self.connection['password'],
                                        username="Administrator")

    def run(self):
        self._connect()

        self._setup_my_slice()


        while True:
            message = self.in_queue.get()

            if message == STOP:
                self.out_queue.put(message)
                break

            if message == GET_READY or message == ITERATION_DONE:
                self.out_queue.put(message)
                continue

            if message == DELETE:
                self.out_queue.put(message)
                self._delete()

            if message == SET:
                self.out_queue.put(message)
                self._set()

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

    # Setup the workers 'slice' of the total keys in the initial SIZES
    def _setup_my_slice(self):
        items_per_size_class = int(self.num_items/len(SIZES))
        total = 0
        for s in SIZES:
            for i in range(items_per_size_class):
                self._set_with_retry(i, s[0])
                total = total + 1

        # Fill in the gap with final size class
        for i in range(total, self.num_items):
            self._set_with_retry(i, SIZES[-1])

    def _delete(self):
        items_per_size_class = int(self.num_items/len(SIZES))
        items_to_delete = items_per_size_class * 0.10
        for s in SIZES:
            for i in range(int(items_to_delete)):
                self._del_with_retry(i, s[self.iteration])

    def _set(self):
        self.iteration = self.iteration + 1
        if self.iteration >= 8:
            self.iteration = 0

        items_per_size_class = int(self.num_items/len(SIZES))
        items_to_set = items_per_size_class * 0.10
        for s in SIZES:
            for i in range(int(items_to_set)):
                self._set_with_retry(i, s[self.iteration])


class Supervisor(Worker):

    SLEEP_TIME = 1

    def __init__(self,
                 number,
                 partner,
                 connection,
                 queues,
                 in_queue,
                 out_queue,
                 num_items,
                 num_iterations,
                 key_slice):
        super().__init__(number,
                         partner,
                         connection,
                         num_items,
                         in_queue, out_queue, key_slice)
        self.queues = queues
        self.num_items = num_items
        self.num_iterations = num_iterations

    def run(self):
        """Run the Supervisor.

        This is similar to Worker, except that completed documents are not
        added back to the output queue. When the last document is seen as
        completed, a new iteration is started.
        """
        logger.info('Starting KeyFragger supervisor')

        # We defer creating the Couchbase object until we are actually
        # 'in' the separate process here.
        self._connect()

        self.out_queue.put(GET_READY)
        self._setup_my_slice()
        message = self.in_queue.get()
        logger.warn("EVERYONE READY")
        time.sleep(20)

        if message != GET_READY:
            logger.warn("Supervisor start-up expected 2 got {0}".format(message))

        mode = DELETE
        for iteration in range(self.num_iterations):
            if mode == DELETE:
                logger.info('Running delete phase')
                self.out_queue.put(DELETE)
                self._delete()
            if mode == SET:
                logger.info('Running set phase')
                self.out_queue.put(SET)
                self._set()

            message = self.in_queue.get()

            if message != mode:
                logger.warn("Supervisor run expected {0} got {1}".format(mode, message))

            # Loop this message around, when it comes back we know all Workers
            # are done
            self.out_queue.put(ITERATION_DONE)
            message = self.in_queue.get()
            if message != ITERATION_DONE:
                logger.warn("Supervisor run expected NEXT_ITERATION {0} got {1}".format(ITERATION_DONE, message))

            logger.info('Completed iteration {0}/{1}'.format(
                iteration + 1, self.num_iterations))
            logger.info('Sleeping for {}s'.format(self.SLEEP_TIME))
            time.sleep(self.SLEEP_TIME)
            if mode == DELETE:
                mode = SET
            else:
                mode = DELETE

        # If we ended with DEL, finish with a SET phase
        if mode == DEL:
            self.out_queue.put(SET)
            logger.info('Finishing with set phase')
            self._set()

        logger.info('Supervisor issuing STOP')
        self.out_queue.put(STOP)

if __name__ == '__main__':
    # Small smoketest
    KeyFragger(num_items=1000000, num_workers=8, num_iterations=20,
             frozen_mode=True, host='localhost', port=9000,
             bucket='bucket-1', password='asdasd').run()