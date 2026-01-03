from .ShmSubscriberOp import ShmSubscriberOp
from .shm_receiver import ShmSynchronizedBufferReceiver
from .DeviceContext import DeviceContextService
from .XYLookupTableSourceOp import XYLookupTableSourceOp
from .shm_rpc import ParameterRpcClient, ParameterRpcServer
import iceoryx2 as iox2

__all__ = ["ShmSubscriberOp", "DeviceContextService", "XYLookupTableSourceOp", "create_shm_subscriber",
           "ParameterRpcClient", "ParameterRpcServer", ]

def create_shm_subscriber():
    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
    return node, ShmSynchronizedBufferReceiver(node)