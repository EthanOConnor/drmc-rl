"""Shared batched forwards for concurrent rollout collections.

Each model gets one ``InferenceService`` thread. Collection threads register
with the ``InferenceHub`` and score through a per-thread ``CoalescedPolicy``
proxy, which submits its rows and waits on a future. A service flushes its
pending rows as soon as no registered collection could still add one (every
client is waiting on some service), when the fill window expires, or when
``max_rows`` is reached. Scoring is row-independent (no batch normalization or
cross-example attention), and learner sampling is keyed per decision, so a
row's result does not depend on which other rows shared its forward beyond
FP32 kernel-shape rounding.

Failure handling: a forward's exception is set on every future in its batch;
waits time out with a clear error; ``close`` (also used on KeyboardInterrupt)
fails every pending and later request so collection threads unwind promptly.
"""
from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextlib import contextmanager
import threading
import time

import numpy as np


class InferenceClosed(RuntimeError):
    pass


class InferenceHub:
    def __init__(self, *, window=.005, timeout=900., max_rows=512):
        self.cond = threading.Condition()
        self.window, self.timeout, self.max_rows = float(window), float(timeout), int(max_rows)
        self.running = 0  # registered clients not currently waiting on a service
        self.closed = None
        self.services = {}
        self.local = threading.local()
        self.stats = dict(forwards=0, rows=0, requests=0)

    def service(self, policy, name=None):
        """The one service for this policy object, started on first use."""
        with self.cond:
            if self.closed is not None:
                raise self.closed
            service = self.services.get(id(policy))
            if service is None:
                service = self.services[id(policy)] = InferenceService(self, policy, name or type(policy).__name__)
            return service

    def proxy(self, policy, name=None):
        return CoalescedPolicy(self.service(policy, name))

    @contextmanager
    def client(self):
        """Register the calling thread as a collection that may submit rows."""
        with self.cond:
            if self.closed is not None:
                raise self.closed
            self.running += 1
            self.local.registered = True
        try:
            yield
        finally:
            with self.cond:
                self.running -= 1
                self.local.registered = False
                self.cond.notify_all()

    def close(self, error=None):
        with self.cond:
            if self.closed is None:
                self.closed = error if error is not None else InferenceClosed("inference hub closed")
            self.cond.notify_all()
            services = list(self.services.values())
        for service in services:
            service.thread.join(timeout=60)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, trace):
        self.close(InferenceClosed(f"inference hub closed after {kind.__name__}") if kind else None)


class InferenceService:
    def __init__(self, hub, policy, name):
        self.hub, self.policy, self.name = hub, policy, name
        self.pending = []
        self.thread = threading.Thread(target=self._loop, name=f"inference-{name}", daemon=True)
        self.thread.start()

    def submit(self, obs, infos):
        hub = self.hub
        future = Future()
        registered = getattr(hub.local, "registered", False)
        with hub.cond:
            if hub.closed is not None:
                raise hub.closed
            self.pending.append((obs, infos, future))
            if registered:
                hub.running -= 1
            hub.cond.notify_all()
        try:
            return future.result(timeout=hub.timeout)
        except FutureTimeout:
            raise RuntimeError(f"coalesced {self.name} inference did not answer within {hub.timeout:.0f} s") from None
        finally:
            if registered:
                with hub.cond:
                    hub.running += 1

    def _take(self):
        hub = self.hub
        with hub.cond:
            while not self.pending and hub.closed is None:
                hub.cond.wait()
            if hub.closed is not None:
                batch, self.pending = self.pending, []
                return batch, hub.closed
            deadline = time.monotonic() + hub.window
            while hub.closed is None and hub.running > 0 and sum(len(p[1]) for p in self.pending) < hub.max_rows:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                hub.cond.wait(remaining)
            batch, self.pending = self.pending, []
            return batch, hub.closed

    def _loop(self):
        while True:
            batch, closed = self._take()
            if closed is not None:
                for _, _, future in batch:
                    future.set_exception(closed)
                return
            try:
                obs = np.concatenate([p[0] for p in batch])
                infos = [info for p in batch for info in p[1]]
                actions, masks, logits = self.policy.score(obs, infos)
                records = getattr(self.policy, "learning_records", None)
                with self.hub.cond:
                    self.hub.stats["forwards"] += 1
                    self.hub.stats["rows"] += len(infos)
                    self.hub.stats["requests"] += len(batch)
                start = 0
                for _, rows, future in batch:
                    stop = start + len(rows)
                    future.set_result((actions[start:stop], masks[start:stop], logits[start:stop],
                                       None if records is None else records[start:stop]))
                    start = stop
            except BaseException as error:  # noqa: BLE001 - every waiter must learn of it
                for _, _, future in batch:
                    if not future.done():
                        future.set_exception(error)


class CoalescedPolicy:
    """Per-thread stand-in for one policy; attribute reads go to the policy."""

    def __init__(self, service):
        self._service = service
        self.learning_records = None

    def __getattr__(self, name):
        return getattr(self._service.policy, name)

    def score(self, obs, infos):
        actions, masks, logits, records = self._service.submit(obs, infos)
        self.learning_records = records
        return actions, masks, logits
