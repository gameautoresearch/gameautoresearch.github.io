"""Modal sandbox helpers shared by every launcher in this repository.

Waits, read-only RPC retries and provider allocation receipts for the pinned
Modal SDK. This is not a session provider: the Generals research backend's
workers are in harness/games/generals/provider.py. Reference-opponent training
launchers import these helpers and copy this file into their source snapshots,
so it depends only on the standard library, Modal and transport.py.
"""
import asyncio
import time

# Modal work for this repository uses only this environment (AGENTS.md).
ENVIRONMENT='game-autoresearch-project'


async def _poll_wait(process):
    """Pinned SDK workaround: short status RPCs instead of TaskExecWait."""
    while (code:=await process.poll()) is None:await asyncio.sleep(.05)
    return code

def configure_sandbox_waits():
    import modal
    if modal.__version__!='1.5.2':raise RuntimeError('Sandbox transport adapter requires pinned Modal 1.5.2')
    from simulators.generals.autoresearch.transport import install_transport_guard
    install_transport_guard()
    from modal.container_process import _ContainerProcess
    # SDK filesystem helpers await this internal method after mkdir/read/write.
    # No command is re-executed: poll observes the same exec ID and deadline.
    _ContainerProcess.wait=_poll_wait

def wait_process(process):
    while (code:=read_rpc(process.poll)) is None:time.sleep(.05)
    return code

def reset_rpc_channels():
    """Discard stale transports on the SDK event loop; retain client identity."""
    async def reset():
        from modal.client import _Client
        client=await _Client.from_env()
        client._connection_manager.close()
    from modal._utils.async_utils import synchronizer
    synchronizer.create_blocking(reset)()

def read_rpc(call):
    """Retry the observed closed-channel race, without starting another allocation.

    Only read-only/idempotent status operations use this wrapper. The original
    provider deadline and every failed allocation remain charged.
    """
    from grpclib.exceptions import StreamTerminatedError
    for attempt in range(4):
        try:return call()
        except (AttributeError,StreamTerminatedError) as exc:
            if (isinstance(exc,AttributeError) and "'Connection' object has no attribute '_transport'" not in str(exc)) or attempt==3:raise
            reset_rpc_channels()
            time.sleep(.2*(attempt+1))

def allocation_info(app_id,sandbox_id):
    """Pinned Modal 1.5.2 provider metadata; no guest-reported clock is trusted."""
    async def fetch():
        from modal.client import _Client
        from modal_proto import api_pb2
        client=await _Client.from_env()
        before=None
        while True:
            request=api_pb2.SandboxListRequest(app_id=app_id,environment_name=ENVIRONMENT,include_finished=True,
                **({'before_timestamp':before} if before is not None else {}))
            response=await client.stub.SandboxList(request)
            for item in response.sandboxes:
                if item.id==sandbox_id:
                    task=item.task_info
                    return {'provider_id':sandbox_id,'task_id':task.id,'created_at':item.created_at,
                        'started_at':task.started_at,'finished_at':task.finished_at,'gpu_type':task.gpu_type}
            if not response.sandboxes:return None
            before=response.sandboxes[-1].created_at
    # Use the SDK's own event loop: asyncio.run() would reuse a cached client and
    # gRPC channel belonging to the synchronizer's different loop.
    from modal._utils.async_utils import synchronizer
    return read_rpc(synchronizer.create_blocking(fetch))
