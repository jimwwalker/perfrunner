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


SIZES = (8,
         32,
         80,
         128,
         224)

# A small sized and fixed value, ideally values never get allocated into a
# bucket we're churning with keys
VALUE_SIZE = 8
VALUE = bytearray(121 for _ in range(VALUE_SIZE))

class AlwaysPromote:

    """Promotion policy for baseline case - always promote."""

    def __init__(self, max_size):
        self.max_size = max_size

    def build_generator(self):
        return SequenceIterator(self.max_size)


class Freeze:

    def __init__(self, num_items, num_iterations, max_size):
        self.num_items = num_items
        # Initialize deterministic pseudo-RNG for when to freeze docs.
        self.rng = random.Random(0)
        self.lock = multiprocessing.Lock()
        # Aim to have 10% of documents left at the end.
        self.freeze_probability = self._calc_freeze_probability(num_iterations=num_iterations,
                                                                final_fraction=0.2)
        self.max_size = max_size

    def build_generator(self):
        """Return a sequence of sizes which ramps from minimum to maximum size.

        If 'freeze' is true, then freeze the sequence at a random position, i.e.
        don't ramp all the way up to max_size.
        """
        if self.rng.random() < self.freeze_probability:
            return SequenceIterator(self.rng.choice(SIZES[:SIZES.index(self.max_size)]))
        else:
            return SequenceIterator(self.max_size)

    def _calc_freeze_probability(self, num_iterations, final_fraction):
        """Return the freeze probability (per iteration)."""
        return 1.0 - (final_fraction ** (1.0 / num_iterations))


class KeyFragger:

    def __init__(self, num_items, num_workers, num_iterations, frozen_mode,
                 host, port, bucket, password):

        # See comment above SIZES list as to why we don't run bigger
        # The test will try and write num_items * len(SIZES) keys...
        assert num_items <= 9999999

        self.num_items = num_items
        self.num_workers = num_workers

        # 5000 items, 20 workers, 6 SIZES
        # 250 per worker
        # 41 per class,
        items_per_worker = int(num_items / num_workers)

        max_size = SIZES[-1]
        if frozen_mode:
            promotion_policy = Freeze(num_items, num_iterations, max_size)
        else:
            promotion_policy = AlwaysPromote(max_size)

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
                               promotion_policy=promotion_policy,
                               num_items=items_per_worker,
                               num_iterations=num_iterations,
                               max_size=max_size,
                               key_slice=(start, num_items))
            else:
                t = Worker(number=i,
                           partner=partner,
                           connection=connection,
                           num_items=items_per_worker,
                           in_queue=self.queues[i],
                           out_queue=self.queues[(i + 1) % self.num_workers],
                           promotion_policy=promotion_policy,
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


class Worker(multiprocessing.Process):

    def __init__(self,
                 number,
                 partner,
                 connection,
                 num_items,
                 in_queue,
                 out_queue,
                 promotion_policy,
                 key_slice):
        super().__init__()
        self.id = number
        self.partner = partner
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.promotion_policy = promotion_policy
        self.connection = connection
        self.key_slice = key_slice
        self.num_items = num_items

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

    def run1(self):
        """Run a Worker.

        They run essentially forever, taking document size iterators from the
        input queue and adding them to the output queue for the next guy.
        """
        # We defer creating the Couchbase object until we are actually 'in' the
        # separate process here.
        self._connect()

        self._set_my_key_slice()
        # Send ready message
        self.out_queue.put((-2, None, -2))

        while True:
            next_size = None
            (i, doc, size) = self.in_queue.get()
            # We use a "magic" null generator to terminate the workers
            if not doc:
                # Pass the message on...
                self.out_queue.put((i, doc, size))

                if i > 0 or i < -2:
                    logger.warn('Worker-{0}: received unknown stop message {1}'.format(self.start, i))

                # i = 0 or -2, go back to queue.get
                if i == 0 or i == -2:
                    continue

                # i = -1, done, but set my slice first
                if i == -1:
                    self._set_my_key_slice()

                break
            # Actually perform the del/set.
            try:
                pre_size  = doc.previous()
                next_size = doc.next()

                # Delete the previous size class key
                if pre_size:
                    self._del_with_retry(str(i), pre_size)

                self._set_with_retry(str(i), next_size)
                size = next_size
            except StopIteration:
                pass
            self.out_queue.put((i, doc, size))

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
                self._set_with_retry(i, s)
                total = total + 1

        # Fill in the gap with final size class
        for i in range(total, self.num_items):
            self._set_with_retry(i, SIZES[-1])

class SequenceIterator:
    def __init__(self, max_size):
        self.sizes = list(SIZES[:SIZES.index(max_size) + 1])
        self.preSize = None

    def next(self):
        if self.sizes:
            self.preSize = self.sizes.pop(0)
            return self.preSize
        else:
            raise StopIteration

    def previous(self):
        return self.preSize


class Supervisor(Worker):

    SLEEP_TIME = 1

    def __init__(self,
                 number,
                 partner,
                 connection,
                 queues,
                 in_queue,
                 out_queue,
                 promotion_policy,
                 num_items,
                 num_iterations,
                 max_size,
                 key_slice):
        super().__init__(number,
                         partner,
                         connection,
                         num_items,
                         in_queue, out_queue, promotion_policy, key_slice)
        self.queues = queues
        self.num_items = num_items
        self.num_iterations = num_iterations
        self.max_size = max_size

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

        self._setup_my_slice()

        return

        # Wait to receive the ready message from workers, this means everyon
        # has populate their key-slice and server memory should be stable
        (i, _, _) = self.in_queue.get()
        if i != -2:
            logger.warn("Supervisor start-up expected -2 got {0}".format(i))

        logger.info('All workers have populated')
        time.sleep(30)
        logger.info('All workers have populated')
        # Create initial list of documents on the 'finished' queue
        finished_items = list()
        for i in range(self.num_items):
            finished_items.append((i, 0))

        for iteration in range(self.num_iterations):

            # Create a tuple for each item in the finished queue, of
            # (doc_id, generator, doc_size). For the first iteration
            # this will be all items, for subsequent iterations it may
            # be fewer if some have been frozen.
            # Spread these across the input queues of all workers, to ensure
            # that each worker operates on different sizes.
            expected_items = len(finished_items)
            num_queues = len(self.queues)
            for (i, size) in list(finished_items):
                queue_index = i % num_queues
                self.queues[queue_index].put(
                    (i,
                     self.promotion_policy.build_generator(),
                     0))
            finished_items = list()

            while expected_items > 0:
                (i, doc, size) = self.in_queue.get()
                # Ignore the extra ready messages of which we should get one per
                # worker
                if not doc:
                    if i != -2:
                        logger.warn("Supervisor running expected -2 got {0}".format(i))
                    continue
                try:
                    pre_size = doc.previous()

                    # Delete the previous size class key
                    if pre_size:
                        self._del_with_retry(str(i), pre_size)

                    next_size = doc.next()
                    self._set_with_retry(str(i), next_size)
                    size = next_size
                    self.out_queue.put((i, doc, size))
                except StopIteration:
                    # Note: Items are not put back on out_queue at end of an
                    # iteration (unlike Worker), instead we keep for the next
                    # iteration, to build the new generators.
                    finished_items.append((i, size))
                    if len(finished_items) == expected_items:
                        # Got all items, end of iteration.
                        break

            assert self.in_queue.empty()
            assert self.out_queue.empty()

            # Any finished items which didn't reach max size should be
            # removed from the next iteration - we want to leave them
            # frozen at their last size.
            finished_items = [(ii, sz) for (ii, sz) in finished_items if sz == self.max_size]

            logger.info('Completed iteration {}/{}, frozen {}/{} documents (aggregate)'.format(
                iteration + 1, self.num_iterations,
                self.num_items - len(finished_items), self.num_items))
            # Sleep at end of iteration to give disk write queue chance to drain.
            logger.info('Sleeping for {}s'.format(self.SLEEP_TIME))
            time.sleep(self.SLEEP_TIME)

        # All iterations complete.

        # Shutdown in two passes, first pass ensures that workers stop set/del
        # Second pass triggers them to just set their key slice and end

        # First send a special None, 0 message to get workers to stop work
        self.out_queue.put((0, None, 0))
        (i, doc, _) = self.in_queue.get()

        # All workers read the message as we have received it
        if not doc:
            if i != 0:
                logger.warn('Supervisor shutdown expected 0 got {0}'.format(i))
            # Second send a special None, -1 message to get workers to finalise
            # their key-slice and shutdown
            self.out_queue.put((-1, None, -1))
        else:
           logger.warn('Supervisor shutdown expected None doc got doc:{0}'.format(doc))

        (i, _, _) = self.in_queue.get()
        if i != -1:
            logger.warn('Supervisor shutdown expected -1 got {0}'.format(i))
        self._set_my_key_slice()

if __name__ == '__main__':
    # Small smoketest
    # num_items * len(SIZES) is how many keys will be generated
    KeyFragger(num_items=10000, num_workers=8, num_iterations=10,
             frozen_mode=True, host='localhost', port=9000,
             bucket='bucket-1', password='asdasd').run()