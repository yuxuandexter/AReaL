"""Staleness-aware capacity manager for rollout generation.

This module provides the StalenessManager class which manages capacity
and staleness constraints for asynchronous rollout generation in RL training.
"""

from threading import Lock
from collections import deque
import statistics

from areal.api.io_struct import RolloutStat


class StalenessManager:
    """Manages rollout capacity based on staleness and concurrency constraints.

    The manager ensures that:
    1. The number of concurrent rollouts doesn't exceed the configured maximum
    2. Rollouts don't become too stale (off-policy) by limiting acceptance based on
       the current model version and maximum allowed offpolicyness

    Parameters
    ----------
    max_concurrent_rollouts : int
        Maximum number of concurrent rollouts allowed
    consumer_batch_size : int
        Expected batch size for consuming rollouts during training
    max_staleness : int
        Maximum allowed offpolicyness (version difference) for rollouts
    tracker_queue_size : int | None, optional
        Size of the sliding window for tracking average request length.
        If None, defaults to max_concurrent_rollouts.
    """

    def __init__(
        self,
        max_concurrent_rollouts: int,
        consumer_batch_size: int,
        max_staleness: int,
        tracker_queue_size: int | None = None,
    ):
        """Initialize the staleness manager.

        Parameters
        ----------
        max_concurrent_rollouts : int
            Maximum number of concurrent rollouts allowed
        consumer_batch_size : int
            Expected batch size for consuming rollouts during training
        max_staleness : int
            Maximum allowed offpolicyness (version difference) for rollouts
        tracker_queue_size : int | None, optional
            Size of the sliding window for tracking average request length.
            If None, defaults to max_concurrent_rollouts.
        """
        self.max_concurrent_rollouts = max_concurrent_rollouts
        self.consumer_batch_size = consumer_batch_size
        self.max_staleness = max_staleness

        # Thread-safe access to rollout statistics
        self.lock = Lock()
        self.rollout_stat = RolloutStat()

        # For tracking average request length
        if tracker_queue_size is None:
            tracker_queue_size = max(1, max_concurrent_rollouts)
        self.request_lengths = deque(maxlen=tracker_queue_size)

    def get_pending_limit(self) -> int:
        """Get the maximum number of pending rollouts allowed.

        Returns
        -------
        int
            Maximum number of pending rollouts (enqueued)
        """
        return (self.max_staleness + 1) * self.consumer_batch_size

    def get_capacity(self, current_version: int) -> int:
        """Calculate available capacity for new rollouts.

        This method considers both concurrency limits and staleness constraints
        to determine how many new rollouts can be accepted.

        The capacity calculation ensures:
        1. The number of running rollouts doesn't exceed max_concurrent_rollouts
        2. Samples don't become too stale by limiting based on:
           - current_version: The current model version
           - max_staleness: Maximum allowed version difference
           - consumer_batch_size: Expected batch size for training

        Parameters
        ----------
        current_version : int
            The current version of the model weights

        Returns
        -------
        int
            Number of new rollout slots available. Can be negative if over capacity.

        Notes
        -----
        The staleness control formula is:
        max_samples = (max_staleness + current_version + 1) * consumer_batch_size
        capacity = min(concurrency_limit, max_samples - current_samples)

        This ensures that by the time samples are consumed, they won't exceed
        the maximum allowed staleness.
        """
        with self.lock:
            # Calculate concurrency-based capacity
            max_concurrent_rollouts = max(1, self.max_concurrent_rollouts)
            concurrency_capacity = max_concurrent_rollouts - self.rollout_stat.running

            # Calculate staleness-based capacity
            ofp = self.max_staleness
            sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
            consumer_bs = max(1, self.consumer_batch_size)
            staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt

            # Return the minimum of both constraints
            capacity = min(concurrency_capacity, staleness_capacity)
            return capacity

    def on_rollout_enqueued(self) -> None:
        """Callback when a rollout is enqueued as a pending input task.

        Thread-safe method to increment the enqueued counters.
        """
        with self.lock:
            self.rollout_stat.enqueued += 1

    def on_rollout_submitted(self, request_length: float = 0) -> None:
        """Callback when a rollout is submitted for execution.

        Thread-safe method to decrement enqueued counter and increment running counters.

        Parameters
        ----------
        request_length : float, optional
            Length of the submitted request (e.g. max_new_tokens) for tracking.
            Default is 0.
        """
        with self.lock:
            self.rollout_stat.enqueued -= 1
            self.rollout_stat.running += 1
            if request_length > 0:
                self.request_lengths.append(request_length)

    def get_request_length_stats(self) -> dict[str, float]:
        """Get statistics of recent request lengths.

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - avg: Average request length
            - std: Standard deviation of request lengths
            - top25_mean: Mean of the top 25% longest requests
        """
        with self.lock:
            if not self.request_lengths:
                return {"avg": 0.0, "std": 0.0, "top25_mean": 0.0}

            data = list(self.request_lengths)
            avg = statistics.mean(data)
            std = statistics.stdev(data) if len(data) > 1 else 0.0

            # Calculate mean of top 25%
            data.sort(reverse=True)
            # Ensure at least 1 item is selected if list is not empty
            k = max(1, int(len(data) * 0.25))
            top_k = data[:k]
            top25_mean = statistics.mean(top_k)

            return {"avg": avg, "std": std, "top25_mean": top25_mean}

    def on_rollout_accepted(self) -> None:
        """Callback when a rollout completes successfully and is accepted.

        Thread-safe method to decrement running counter and increment accepted counter.
        """
        with self.lock:
            self.rollout_stat.running -= 1
            self.rollout_stat.accepted += 1

    def on_rollout_rejected(self) -> None:
        """Callback when a rollout completes but is rejected.

        Thread-safe method to decrement running counter and increment rejected counter.
        """
        with self.lock:
            self.rollout_stat.running -= 1
            self.rollout_stat.rejected += 1

    def get_stats(self) -> RolloutStat:
        """Get a snapshot of current rollout statistics.

        Returns
        -------
        RolloutStat
            Current rollout statistics (enqueued, accepted, running)
        """
        with self.lock:
            return RolloutStat(
                accepted=self.rollout_stat.accepted,
                enqueued=self.rollout_stat.enqueued,
                rejected=self.rollout_stat.rejected,
                running=self.rollout_stat.running,
            )
