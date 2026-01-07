import asyncio

class AtomicInteger():
    def __init__(self, initial: int = 0):
        self._count = initial
        self.lock = asyncio.Lock()

    async def get(self):
        async with self.lock:
            return self._count

    async def set(self, value: int):
        async with self.lock:
            self._count = value

    async def increment(self):
        async with self.lock:
            self._count += 1
    
    async def decrement(self):
        async with self.lock:
            self._count -= 1

    async def increment_and_get(self):
        await self.increment()
        return await self.get()
    
    async def decrement_and_get(self):
        await self.decrement()
        return await self.get()