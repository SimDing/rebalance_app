import asyncio
import contextlib
from pytr.api import TradeRepublicApi

class ApiWrapper:
    def __init__(self, api: TradeRepublicApi):
        self._api = api
        self._pending = {}
        self._listening = False

    async def _recv_loop(self, loop):
        while True:
            subscription_id, _, payload = await self._api.recv()
            loop.create_task(self._api.unsubscribe(subscription_id))
            fut = self._pending.pop(subscription_id, None)
            if fut and not fut.done():
                fut.set_result(payload)
                continue
            print(f"Unhandled message: {subscription_id}, {payload}")
    
    def _start_listening(self, loop):
        if not self._listening:
            self._task = loop.create_task(self._recv_loop(loop))
            self._listening = True

    async def request(self, send_coro):
        subscription_id = await send_coro
        loop = asyncio.get_running_loop()
        self._start_listening(loop)
        fut = loop.create_future()
        self._pending[subscription_id] = fut
        return await fut
    
    async def close(self):
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

