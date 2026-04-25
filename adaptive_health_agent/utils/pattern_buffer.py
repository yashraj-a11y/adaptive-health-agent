"""
Pattern Buffer Module

Maintains counters for consecutive anomalous metric readings.
A pattern is "confirmed" when a metric's counter reaches or exceeds
PATTERN_BUFFER_THRESHOLD (default 3), indicating a sustained anomaly
rather than a transient spike.
"""

import os
from dotenv import load_dotenv

load_dotenv()

PATTERN_BUFFER_THRESHOLD = int(os.getenv("PATTERN_BUFFER_THRESHOLD", 3))


class PatternBuffer:
    """Tracks consecutive anomalous readings for each monitored metric.

    Supported metrics:
        hr_elevated, hrv_low, stress_elevated, spo2_low,
        temp_elevated, breathing_elevated, recovery_low
    """

    def __init__(self):
        """Initialize all metric counters to zero."""
        self._counters = {
            "hr_elevated": 0,
            "hrv_low": 0,
            "stress_elevated": 0,
            "spo2_low": 0,
            "temp_elevated": 0,
            "breathing_elevated": 0,
            "recovery_low": 0,
            "sleep_efficiency_low": 0,
        }

    def increment(self, metric: str) -> None:
        """Increment the counter for a specific metric.

        Args:
            metric: The metric name (must be one of the tracked metrics).
        """
        if metric in self._counters:
            self._counters[metric] += 1
        else:
            print(f"[Pattern Buffer] Unknown metric: {metric}")

    def reset(self, metric: str) -> None:
        """Reset the counter for a specific metric to zero.

        Args:
            metric: The metric name to reset.
        """
        if metric in self._counters:
            self._counters[metric] = 0

    def is_confirmed(self, metric: str) -> bool:
        """Check if a metric's pattern is confirmed (counter >= threshold).

        Args:
            metric: The metric name to check.

        Returns:
            bool: True if the counter has reached the threshold.
        """
        if metric in self._counters:
            return self._counters[metric] >= PATTERN_BUFFER_THRESHOLD
        return False

    def get_confirmed_patterns(self) -> list:
        """Return a list of all metrics that have confirmed patterns.

        Returns:
            list: List of metric names where counter >= threshold.
        """
        return [
            metric for metric, count in self._counters.items()
            if count >= PATTERN_BUFFER_THRESHOLD
        ]

    def reset_confirmed(self, metric: str) -> None:
        """Reset a confirmed pattern's counter with cooldown.

        Sets to -5 so it takes 8+ more consecutive deviations to re-confirm.
        Prevents rapid re-triggering of the same pattern.

        Args:
            metric: The metric name to reset after confirmation processing.
        """
        if metric in self._counters:
            self._counters[metric] = -5  # Cooldown: needs 8 more consecutive to re-confirm

    def get_count(self, metric: str) -> int:
        """Return the current counter value for a metric.

        Args:
            metric: The metric name.

        Returns:
            int: Current counter value, or 0 if metric is unknown.
        """
        return self._counters.get(metric, 0)

    def get_all_counts(self) -> dict:
        """Return a copy of all current counter values.

        Returns:
            dict: All metric counters.
        """
        return dict(self._counters)
