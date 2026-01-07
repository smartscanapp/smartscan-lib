from abc import ABC
from typing import Generic
from smartscan.processor.metrics import MetricsFailure, MetricsSuccess
from smartscan.processor.types import Input, Output


class ProcessorListener(ABC, Generic[Input, Output]):
    async def on_active(self):
        pass
    # made async with asyncio.to_thread
    async def on_progress(self, progress: float):
        pass
    async def on_complete(self, result: MetricsSuccess):
        pass
    async def on_batch_complete(self, batch: list[Output]):
        pass
    async def on_error(self, e: Exception, item: Input):
        pass
    async def on_fail(self, result: MetricsFailure):
        pass