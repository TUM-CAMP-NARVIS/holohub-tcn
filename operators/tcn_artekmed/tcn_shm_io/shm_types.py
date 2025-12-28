import ctypes
from enum import IntEnum


class PubSubEvent(IntEnum):
    """
    Python equivalent of:

    enum class PubSubEvent : uint8_t { ... };
    """

    PublisherConnected = 0
    PublisherDisconnected = 1
    SubscriberConnected = 2
    SubscriberDisconnected = 3
    SentSample = 4
    ReceivedSample = 5
    SentHistory = 6
    Unknown = 7


class ShmSerializedMessage(ctypes.Structure):
    """
    ctypes binding for:

    struct ShmSerializedMessage {
        static inline const std::size_t payloadMaxSize{16384};
        std::array<unsigned char, payloadMaxSize> payload{};
        std::size_t payloadSize{0};
    };
    """

    payloadMaxSize = 16384

    _fields_ = [
        ("payload", ctypes.c_ubyte * payloadMaxSize),
        ("payloadSize", ctypes.c_size_t),
    ]

    def payload_bytes(self) -> bytes:
        """Return the valid payload as bytes (uses payloadSize)."""
        n = int(self.payloadSize)
        if n < 0:
            n = 0
        if n > self.payloadMaxSize:
            n = self.payloadMaxSize
        return bytes(self.payload[:n])

    def __str__(self) -> str:
        return (
            "ShmSerializedMessage { "
            f"payloadSize: {int(self.payloadSize)}"
            " }"
        )

    @staticmethod
    def type_name() -> str:
        """System-wide unique type name required for communication."""
        return "ShmSerializedMessage"

class ShmSerializedStreamHeader(ctypes.Structure):
    """
    ctypes binding for:

    struct ShmSerializedStreamHeader {
        static inline const std::size_t payloadMaxSize{1024};
        std::array<unsigned char, payloadMaxSize> payload{};
        std::size_t payloadSize{0};
        uint64_t timestamp{0};
    };
    """

    payloadMaxSize = 1024

    # Important:
    # - std::array<unsigned char, 1024>  -> c_ubyte * 1024
    # - std::size_t                      -> c_size_t (matches platform pointer width)
    # - uint64_t                         -> c_uint64
    _fields_ = [
        ("payload", ctypes.c_ubyte * payloadMaxSize),
        ("payloadSize", ctypes.c_size_t),
        ("timestamp", ctypes.c_uint64),
    ]

    def payload_bytes(self) -> bytes:
        """Return the valid payload as bytes (uses payloadSize)."""
        n = int(self.payloadSize)
        if n < 0:
            n = 0
        if n > self.payloadMaxSize:
            n = self.payloadMaxSize
        return bytes(self.payload[:n])

    def __str__(self) -> str:
        return (
            "ShmSerializedStreamHeader { "
            f"payloadSize: {int(self.payloadSize)}, "
            f"timestamp: {int(self.timestamp)}"
            " }"
        )

    @staticmethod
    def type_name() -> str:
        """System-wide unique type name required for communication."""
        return "ShmSerializedStreamHeader"