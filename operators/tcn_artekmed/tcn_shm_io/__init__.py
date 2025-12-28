from .ShmSubscriberOp import ShmSubscriberOp
from .shm_receiver import ShmSynchronizedBufferReceiver
from .DeviceContext import DeviceContextService
from .XYLookupTableSourceOp import XYLookupTableSourceOp
import iceoryx2 as iox2

__all__ = ["ShmSubscriberOp", "DeviceContextService", "XYLookupTableSourceOp", "create_shm_subscriber"]

def create_shm_subscriber():
    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
    return node, ShmSynchronizedBufferReceiver(node)