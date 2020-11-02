import ray
import asyncio
import time

ray.init()

@ray.remote
class AsyncActor:
    # multiple invocation of this method can be running in
    # the event loop at the same time
    async def run_concurrent(self, i):
        print("started ", i)
        await asyncio.sleep(2) # concurrent workload here
        print("finished ", i)

actor = AsyncActor.remote()

# regular ray.get
ray.get([actor.run_concurrent.remote(i) for i in range(4)])

@ray.remote
def run_concurrent(i):
    print("started ", i)
    time.sleep(2) # concurrent workload here
    print("finished ", i)

# regular ray.get
ray.get([run_concurrent.remote(i) for i in range(4)])