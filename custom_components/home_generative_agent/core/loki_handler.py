"""Loki and InfluxDB handlers for remote logging and metrics."""

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp

_LOGGER = logging.getLogger(__name__)


class LokiFileBuffer:
    """Fallback file buffer for network failures."""

    def __init__(self, buffer_path: str, max_size_mb: int = 50):
        """Initialize file buffer.

        Args:
            buffer_path: Directory path for buffer files
            max_size_mb: Maximum buffer size in megabytes
        """
        self.buffer_path = Path(buffer_path)
        self.buffer_file = self.buffer_path / "buffer.json"
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure buffer directory exists."""
        try:
            self.buffer_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            _LOGGER.error(f"Failed to create buffer directory: {e}")

    async def append_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Append log entries to buffer file.

        Args:
            entries: List of log entries to buffer
        """
        try:
            # Check size before appending
            if self.buffer_file.exists():
                size = self.buffer_file.stat().st_size
                if size > self.max_size_bytes:
                    _LOGGER.warning(
                        f"Buffer file exceeds {self.max_size_bytes / 1024 / 1024}MB, "
                        "rotating old entries"
                    )
                    await self._rotate()

            # Append entries
            async with aiofiles.open(self.buffer_file, "a") as f:
                for entry in entries:
                    await f.write(json.dumps(entry) + "\n")

        except Exception as e:
            _LOGGER.error(f"Failed to write to buffer file: {e}")

    async def read_and_clear(self) -> List[Dict[str, Any]]:
        """Read all buffered entries and clear the file.

        Returns:
            List of buffered log entries
        """
        entries = []

        if not self.buffer_file.exists():
            return entries

        try:
            async with aiofiles.open(self.buffer_file, "r") as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            _LOGGER.warning(
                                f"Skipping invalid JSON in buffer: {line[:100]}"
                            )

            # Clear the file after successful read
            async with aiofiles.open(self.buffer_file, "w") as f:
                await f.write("")

        except Exception as e:
            _LOGGER.error(f"Failed to read buffer file: {e}")

        return entries

    async def _rotate(self) -> None:
        """Rotate buffer file by keeping only recent half."""
        try:
            entries = []
            async with aiofiles.open(self.buffer_file, "r") as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(line)
                        except Exception:
                            pass

            # Keep only the most recent half
            if entries:
                keep_count = len(entries) // 2
                async with aiofiles.open(self.buffer_file, "w") as f:
                    for entry in entries[-keep_count:]:
                        await f.write(entry + "\n")

        except Exception as e:
            _LOGGER.error(f"Failed to rotate buffer: {e}")


class _BatchHandler:
    """Base class for batched async push/write handlers."""

    def __init__(self, batch_size: int, batch_interval: int):
        """Initialize shared batch handler fields.

        Args:
            batch_size: Number of items to batch before flush
            batch_interval: Seconds to wait before flushing partial batch
        """
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.queue: deque = deque()
        self.batch_task: Optional[asyncio.Task] = None
        self.running = False
        self.consecutive_failures = 0

    async def start(self) -> None:
        """Start the background batch loop task."""
        if not self.running:
            self.running = True
            self.batch_task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        """Stop the background task and flush remaining items."""
        self.running = False
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass

        # Flush remaining items
        if self.queue:
            await self._flush(list(self.queue))

    async def _batch_loop(self) -> None:
        """Background task that flushes batches periodically."""
        while self.running:
            try:
                await asyncio.sleep(self.batch_interval)

                if self.queue:
                    batch = []
                    while self.queue and len(batch) < self.batch_size:
                        batch.append(self.queue.popleft())

                    if batch:
                        await self._flush(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                _LOGGER.error(f"Error in batch loop: {e}")

    async def _flush(self, batch: List[Dict[str, Any]]) -> None:
        """Flush a batch of items. Must be overridden by subclasses.

        Args:
            batch: List of items to flush
        """
        raise NotImplementedError


class LokiHttpHandler(_BatchHandler):
    """Async HTTP handler for pushing logs to Loki."""

    def __init__(
        self,
        loki_url: str,
        buffer_path: str,
        timeout: int = 5,
        batch_size: int = 50,
        batch_interval: int = 10,
    ):
        """Initialize Loki HTTP handler.

        Args:
            loki_url: Loki server URL (e.g., http://192.168.1.10:3100)
            buffer_path: Path for fallback buffer
            timeout: HTTP request timeout in seconds
            batch_size: Number of logs to batch before push
            batch_interval: Seconds to wait before pushing partial batch
        """
        super().__init__(batch_size, batch_interval)
        self.loki_url = loki_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.buffer = LokiFileBuffer(buffer_path)
        self.last_push_time = time.time()

    async def start(self) -> None:
        """Start the background batch push task."""
        await super().start()
        if self.running:
            _LOGGER.info("Loki handler started")

            # Try to send any buffered logs
            await self._retry_buffered_logs()

    async def stop(self) -> None:
        """Stop the background task and flush remaining logs."""
        await super().stop()
        _LOGGER.info("Loki handler stopped")

    async def queue_log(
        self, log_entry: Dict[str, Any], labels: Dict[str, str]
    ) -> None:
        """Queue a log entry for batched push.

        Args:
            log_entry: Log entry dictionary
            labels: Loki labels for the log stream
        """
        self.queue.append({"entry": log_entry, "labels": labels})

    async def _flush(self, batch: List[Dict[str, Any]]) -> None:
        """Push a batch of logs to Loki.

        Args:
            batch: List of log entries with labels
        """
        if not batch:
            return

        try:
            payload = self._format_loki_payload(batch)

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.loki_url}/loki/api/v1/push",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 204:
                        self.consecutive_failures = 0
                        _LOGGER.debug(f"Successfully pushed {len(batch)} logs to Loki")
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Loki returned {response.status}: {error_text}"
                        )

        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            _LOGGER.warning(
                f"Loki push timeout (attempt {self.consecutive_failures}), buffering to file"
            )
            await self._save_to_fallback_buffer(batch)

        except aiohttp.ClientError as e:
            self.consecutive_failures += 1
            _LOGGER.warning(
                f"Loki push failed (attempt {self.consecutive_failures}): {e}, buffering to file"
            )
            await self._save_to_fallback_buffer(batch)

        except Exception as e:
            self.consecutive_failures += 1
            _LOGGER.error(f"Unexpected error pushing to Loki: {e}")
            await self._save_to_fallback_buffer(batch)

    def _format_loki_payload(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format batch into Loki push API format.

        Args:
            batch: List of log entries with labels

        Returns:
            Loki API payload dictionary
        """
        # Group logs by label set
        streams: Dict[str, List[List[str]]] = {}

        for item in batch:
            entry = item["entry"]
            labels = item["labels"]

            # Create label string (e.g., {job="test",level="info"})
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            label_key = "{" + label_str + "}"

            if label_key not in streams:
                streams[label_key] = []

            # Timestamp in nanoseconds
            timestamp = entry.get("timestamp", datetime.utcnow().isoformat() + "Z")
            # Convert ISO8601 to nanosecond timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                ts_ns = str(int(dt.timestamp() * 1e9))
            except Exception:
                ts_ns = str(int(time.time() * 1e9))

            # Format as JSON string for Loki
            log_line = json.dumps(entry)
            streams[label_key].append([ts_ns, log_line])

        # Format into Loki payload
        payload = {
            "streams": [
                {
                    "stream": dict(
                        label.strip("{}").split("=") for label in labels.split(",")
                    ),
                    "values": values,
                }
                for labels, values in streams.items()
            ]
        }

        return payload

    async def _save_to_fallback_buffer(self, batch: List[Dict[str, Any]]) -> None:
        """Save failed batch to file buffer.

        Args:
            batch: List of log entries that failed to push
        """
        try:
            entries = [item["entry"] for item in batch]
            await self.buffer.append_entries(entries)
            _LOGGER.debug(f"Buffered {len(entries)} logs to file")
        except Exception as e:
            _LOGGER.error(f"Failed to save to buffer: {e}")

    async def _retry_buffered_logs(self) -> None:
        """Attempt to send buffered logs."""
        try:
            buffered = await self.buffer.read_and_clear()
            if buffered:
                _LOGGER.info(f"Retrying {len(buffered)} buffered logs")

                # Re-queue with default labels
                for entry in buffered:
                    labels = {
                        "job": "home_generative_agent",
                        "level": entry.get("level", "info"),
                        "source": "buffer",
                    }
                    await self.queue_log(entry, labels)

        except Exception as e:
            _LOGGER.error(f"Failed to retry buffered logs: {e}")


class InfluxMetricsHandler(_BatchHandler):
    """Async handler for pushing metrics to InfluxDB."""

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        batch_size: int = 50,
        batch_interval: int = 10,
    ):
        """Initialize InfluxDB metrics handler.

        Args:
            url: InfluxDB server URL
            token: InfluxDB API token
            org: InfluxDB organization
            bucket: InfluxDB bucket name
            batch_size: Number of metrics to batch
            batch_interval: Seconds between batch writes
        """
        super().__init__(batch_size, batch_interval)
        self.url = url.rstrip("/")
        self.token = token
        self.org = org
        self.bucket = bucket
        self.failed_writes = 0
        self.timeout = aiohttp.ClientTimeout(total=5)

    async def start(self) -> None:
        """Start the background batch write task."""
        await super().start()
        if self.running:
            _LOGGER.info("InfluxDB handler started")

    async def stop(self) -> None:
        """Stop the background task and flush remaining metrics."""
        await super().stop()
        _LOGGER.info("InfluxDB handler stopped")

    async def queue_metric(
        self, measurement: str, fields: Dict[str, float], tags: Dict[str, str]
    ) -> None:
        """Queue a metric for batched write.

        Args:
            measurement: Measurement name
            fields: Field key-value pairs
            tags: Tag key-value pairs
        """
        self.queue.append(
            {
                "measurement": measurement,
                "fields": fields,
                "tags": tags,
                "timestamp": int(time.time() * 1e9),  # nanoseconds
            }
        )

    async def _flush(self, batch: List[Dict[str, Any]]) -> None:
        """Write a batch of metrics to InfluxDB.

        Args:
            batch: List of metric dictionaries
        """
        if not batch:
            return

        try:
            line_protocol = self._format_line_protocol(batch)

            url = f"{self.url}/api/v2/write?org={self.org}&bucket={self.bucket}&precision=ns"
            headers = {
                "Authorization": f"Token {self.token}",
                "Content-Type": "text/plain; charset=utf-8",
            }

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    url, data=line_protocol, headers=headers
                ) as response:
                    if response.status == 204:
                        self.failed_writes = 0
                        _LOGGER.debug(
                            f"Successfully wrote {len(batch)} metrics to InfluxDB"
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"InfluxDB returned {response.status}: {error_text}"
                        )

        except asyncio.TimeoutError:
            self.failed_writes += 1
            _LOGGER.warning(f"InfluxDB write timeout (attempt {self.failed_writes})")
            await self._handle_failure(batch)

        except aiohttp.ClientError as e:
            self.failed_writes += 1
            _LOGGER.warning(
                f"InfluxDB write failed (attempt {self.failed_writes}): {e}"
            )
            await self._handle_failure(batch)

        except Exception as e:
            self.failed_writes += 1
            _LOGGER.error(f"Unexpected error writing to InfluxDB: {e}")
            await self._handle_failure(batch)

    def _format_line_protocol(self, batch: List[Dict[str, Any]]) -> str:
        """Format metrics into InfluxDB line protocol.

        Args:
            batch: List of metric dictionaries

        Returns:
            Line protocol formatted string
        """
        lines = []

        for metric in batch:
            measurement = metric["measurement"]
            tags = metric.get("tags", {})
            fields = metric.get("fields", {})
            timestamp = metric.get("timestamp", int(time.time() * 1e9))

            # Format: measurement,tag1=value1,tag2=value2 field1=value1,field2=value2 timestamp
            tag_str = (
                ",".join(f"{k}={v}" for k, v in sorted(tags.items())) if tags else ""
            )
            field_str = ",".join(
                f"{k}={v}" if isinstance(v, (int, float)) else f'{k}="{v}"'
                for k, v in fields.items()
            )

            if tag_str:
                line = f"{measurement},{tag_str} {field_str} {timestamp}"
            else:
                line = f"{measurement} {field_str} {timestamp}"

            lines.append(line)

        return "\n".join(lines)

    async def _handle_failure(self, batch: List[Dict[str, Any]]) -> None:
        """Handle failed metric writes.

        Args:
            batch: Failed batch of metrics
        """
        # Keep only recent 100 metrics in memory buffer
        if self.failed_writes > 3:
            # Too many failures, start discarding old metrics
            while len(self.queue) > 100:
                self.queue.popleft()
            _LOGGER.warning(
                "InfluxDB failures exceeded threshold, discarding old metrics"
            )
        else:
            # Re-queue for retry
            for metric in batch:
                self.queue.append(metric)
