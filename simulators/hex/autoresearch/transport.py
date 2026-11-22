"""Host-only guard for a closed-connection race in the pinned gRPC client."""
_installed=False

def connection_usable(protocol):
    processor=getattr(protocol,'processor',None)
    connection=getattr(processor,'connection',None)
    return connection is not None and not connection.is_closing()

def install_transport_guard():
    global _installed
    if _installed:return
    from modal._utils.grpc_utils import ModalChannel
    from grpclib.protocol import Connection
    from grpclib.exceptions import StreamTerminatedError
    connected=ModalChannel._connected.fget
    create_stream=Connection.create_stream

    def live(channel):
        if not connected(channel):return False
        return connection_usable(channel._protocol)

    def stream(connection,*args,**kwargs):
        # close() deletes _transport before asyncio delivers connection_lost.
        # Report a normal terminated stream instead of leaking AttributeError.
        if connection.is_closing():raise StreamTerminatedError('Transport closed before request stream creation')
        return create_stream(connection,*args,**kwargs)

    ModalChannel._connected=property(live)
    Connection.create_stream=stream
    _installed=True
