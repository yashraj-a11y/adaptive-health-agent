"""
Telemetry Stream Module

Emits health telemetry packets at a configurable interval.
Accepts pre-built scenario packet lists and yields them one at a time
with a configurable delay between emissions.
"""

import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

TELEMETRY_INTERVAL_SECONDS = int(os.getenv("TELEMETRY_INTERVAL_SECONDS", 5))


def stream_packets(packets: list, interval: int = None):
    """
    Generator that yields telemetry packets at the configured interval.

    Args:
        packets: List of telemetry packet dicts to stream.
        interval: Seconds between packets. Defaults to TELEMETRY_INTERVAL_SECONDS.

    Yields:
        dict: One telemetry packet per interval.
    """
    if interval is None:
        interval = TELEMETRY_INTERVAL_SECONDS

    for i, packet in enumerate(packets):
        yield packet
        # Sleep between packets, but not after the last one
        if i < len(packets) - 1:
            time.sleep(interval)


def format_packet(packet: dict) -> str:
    """Format a telemetry packet as a pretty-printed JSON string."""
    return json.dumps(packet, indent=2, default=str)
