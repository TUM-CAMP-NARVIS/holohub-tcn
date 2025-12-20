import capnp  # pip install pycapnp
import os
from pathlib import Path
from typing import Union

SCHEMA_PATH = (Path(__file__) / Path("schema/shm_synchronized_transport.capnp")).resolve()
PACKAGES_PATH = Path(capnp.__file__).resolve().parent.parent
CANDIDATE_INCLUDE_ROOTS = [
    str(SCHEMA_PATH.parent),
    str(PACKAGES_PATH),
]
IMPORTS = [p for p in CANDIDATE_INCLUDE_ROOTS if os.path.isdir(p)]
shm_transport_schema = capnp.load(str(SCHEMA_PATH), imports=IMPORTS)

def decode_buffer_descriptor(buf: Union[bytes, bytearray, memoryview]):
    """
    Decodes a Cap'n Proto message contained in `buf` into streamtypes.FlatMessage.

    Use from_bytes() for the *standard* (unpacked) Cap'n Proto encoding.
    If your sender uses *packed* encoding, use from_bytes_packed() instead.
    """
    b = bytes(buf)  # ensure contiguous bytes
    try:
        # Unpacked encoding (most common for "raw message bytes" unless explicitly packed)
        return shm_transport_schema.ShmBufferDescriptor.from_bytes(b)
    except Exception:
        # Packed encoding fallback (only if your sender used packed serialization)
        return shm_transport_schema.ShmBufferDescriptor.from_bytes_packed(b)

def decode_shm_buffer_connection_status(buf: Union[bytes, bytearray, memoryview]):
    """
    Decodes a Cap'n Proto message contained in `buf` into streamtypes.FlatMessage.

    Use from_bytes() for the *standard* (unpacked) Cap'n Proto encoding.
    If your sender uses *packed* encoding, use from_bytes_packed() instead.
    """
    b = bytes(buf)  # ensure contiguous bytes
    try:
        # Unpacked encoding (most common for "raw message bytes" unless explicitly packed)
        return shm_transport_schema.ShmBufferConnectionStatus.from_bytes(b)
    except Exception:
        # Packed encoding fallback (only if your sender used packed serialization)
        return shm_transport_schema.ShmBufferConnectionStatus.from_bytes_packed(b)

def decode_shm_device_context(buf: Union[bytes, bytearray, memoryview]):
    """
    Decodes a Cap'n Proto message contained in `buf` into streamtypes.FlatMessage.

    Use from_bytes() for the *standard* (unpacked) Cap'n Proto encoding.
    If your sender uses *packed* encoding, use from_bytes_packed() instead.
    """
    b = bytes(buf)  # ensure contiguous bytes
    try:
        # Unpacked encoding (most common for "raw message bytes" unless explicitly packed)
        return shm_transport_schema.ShmDeviceContext.from_bytes(b)
    except Exception:
        # Packed encoding fallback (only if your sender used packed serialization)
        return shm_transport_schema.ShmDeviceContext.from_bytes_packed(b)
