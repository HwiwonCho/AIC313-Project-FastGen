"""Bounded CPU preparation with checkpoint state at the last consumed batch.

Prepared lookahead is deliberately not checkpointed: on resume it is regenerated
from the last consumed sampler/augmentation RNG state. No worker uses CUDA or RNG
from Python/Torch's global generators. The wrapped batch source owns its RNGs.
"""
import copy
import io
import multiprocessing
import queue
import threading
import time


class ThreadPrefetchBatches:
    def __init__(self, source, count, capacity=2):
        self.source, self.count = source, count
        self.capacity = capacity
        self.consumed_state = copy.deepcopy(source.state_dict())
        self.output = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.worker = None

    def state_dict(self):
        return copy.deepcopy(self.consumed_state)

    def load_state_dict(self, state):
        if self.worker is not None:
            raise RuntimeError("Restore prefetch state before consuming batches")
        self.source.load_state_dict(state)
        self.consumed_state = copy.deepcopy(state)

    def _produce(self):
        while not self.stop.is_set():
            try:
                images, labels = self.source.batch(self.count, "cpu")
                item = (images, labels, self.source.state_dict(), None)
            except BaseException as error:
                item = (None, None, None, repr(error))
            while not self.stop.is_set():
                try:
                    self.output.put(item, timeout=.2)
                    break
                except queue.Full:
                    pass
            if item[-1] is not None:
                return

    def batch(self, count, device):
        if count != self.count:
            raise ValueError("Prefetch uses a fixed microbatch size")
        if self.worker is None:
            self._start()
        deadline = time.monotonic()+60
        while True:
            try:
                images, labels, state, error = self.output.get(timeout=.5)
                break
            except queue.Empty:
                if not self.worker.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("CPU batch preparation stopped or timed out")
        if error:
            raise RuntimeError("CPU batch preparation failed: "+error)
        images, labels = self._decode(images, labels)
        self.consumed_state = state
        # Same ordered CUDA stream and tensors as the original loader. CPU image
        # decoding/augmentation alone is overlapped; no training call is skipped.
        return images.to(device), labels.to(device)

    def _start(self):
        self.worker = threading.Thread(target=self._produce, name="fastgen-data", daemon=True)
        self.worker.start()

    def _decode(self, images, labels):
        return images, labels

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                raise RuntimeError("CPU prefetch worker did not stop")


def _process_producer(source, count, output, stop):
    import torch
    torch.set_num_threads(1)
    # Pending lookahead is regenerated after resume, so the worker must not wait
    # for a full IPC pipe to drain when the consumer closes at a checkpoint.
    output.cancel_join_thread()
    while not stop.is_set():
        try:
            images, labels = source.batch(count, "cpu")
            buffer = io.BytesIO()
            torch.save((images, labels), buffer)
            item = (buffer.getvalue(), None, source.state_dict(), None)
        except BaseException as error:
            item = (None, None, None, repr(error))
        while not stop.is_set():
            try:
                output.put(item, timeout=.2)
                break
            except queue.Full:
                pass
        if item[-1] is not None:
            return


class PrefetchBatches(ThreadPrefetchBatches):
    """One spawn-based CPU worker; no CUDA context and at most two queued batches."""
    def __init__(self, source, count, capacity=2):
        self.source, self.count, self.capacity = source, count, capacity
        self.consumed_state = copy.deepcopy(source.state_dict())
        self.context = multiprocessing.get_context("spawn")
        self.output = self.context.Queue(maxsize=capacity)
        self.stop = self.context.Event()
        self.worker = None

    def _start(self):
        self.worker = self.context.Process(target=_process_producer,
            args=(self.source,self.count,self.output,self.stop),name="fastgen-cpu-data",daemon=True)
        self.worker.start()

    def _decode(self, images, labels):
        import torch
        return torch.load(io.BytesIO(images),map_location="cpu",weights_only=True)

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                self.worker.terminate()
                self.worker.join(timeout=5)
                raise RuntimeError("CPU data process required termination")
        self.output.close()
        self.output.cancel_join_thread()
