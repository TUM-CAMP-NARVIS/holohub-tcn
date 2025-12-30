import re
import ctypes
import logging
import iceoryx2 as iox2

from .shm_types import ShmSerializedMessage, ShmSerializedStreamHeader, PubSubEvent
from .shm_serde import decode_shm_buffer_connection_status, decode_shm_device_context, decode_buffer_descriptor

log = logging.getLogger(__name__)
DEVICE_CONTEXT_MATCH = re.compile(r'^(.+)\/DEVICE_CONTEXT\/SensorCalibration$')

class ShmSynchronizedBufferReceiver:

    def __init__(self, node):
        self.node = node
        self.subscriber_service = None
        self.subscriber = None

    @staticmethod
    def discover_devices():
        camera_names = set()
        services = iox2.Service.list(iox2.config.global_config(), iox2.ServiceType.Ipc)
        for service in services:
            match = DEVICE_CONTEXT_MATCH.match(service.name().to_string())
            if match:
                camera_names.add(match.group(1))
        return list(sorted(camera_names))

    def retrieve_device_context(self, camera_name):
        log.info(f"Receive Camera Device Context: {camera_name}")
        service = (
            self.node.service_builder(iox2.ServiceName.new(f"{camera_name}/DEVICE_CONTEXT/SensorCalibration"))
            .publish_subscribe(ShmSerializedMessage)
            .history_size(1)
            .subscriber_max_buffer_size(4)
            .open()
        )
        event = (
            self.node.service_builder(
                iox2.ServiceName.new(f"{camera_name}/DEVICE_CONTEXT/SensorCalibration")).event().open_or_create()
        )
        subscriber = service.subscriber_builder().create()
        notifier = event.notifier_builder().create()

        notifier.notify_with_custom_event_id(iox2.EventId.new(PubSubEvent.SubscriberConnected))

        result = None
        while True:
            sample = subscriber.receive()
            if sample is not None:
                contents = sample.payload().contents
                log.debug(f"received device_context payload {contents}")
                with decode_shm_device_context(contents.payload_bytes()) as message:
                    result = message.to_dict()
                    log.debug(f"decoded device_context message: {result}")
                notifier.notify_with_custom_event_id(iox2.EventId.new(PubSubEvent.ReceivedSample))
                break

        return result

    def retrieve_channel_config(self, stream_name):
        result = None
        service_name = f"{stream_name}/COMPOSITE_BUFFER/Config"
        log.info(f"retrieve_channel_config({service_name})")
        service = (
            self.node.service_builder(iox2.ServiceName.new(service_name))
            .publish_subscribe(ShmSerializedMessage)
            .history_size(1)
            .subscriber_max_buffer_size(4)
            .open()
        )
        event = (
            self.node.service_builder(
                iox2.ServiceName.new(service_name)).event().open_or_create()
        )
        subscriber = service.subscriber_builder().create()
        notifier = event.notifier_builder().create()

        notifier.notify_with_custom_event_id(iox2.EventId.new(PubSubEvent.SubscriberConnected))

        while True:
            sample = subscriber.receive()
            if sample is not None:
                contents = sample.payload().contents
                log.debug(f"received config payload: {contents}")
                with decode_shm_buffer_connection_status(contents.payload_bytes()) as message:
                    result = message.to_dict()
                    log.debug(f"decoded config message: {result}")
                notifier.notify_with_custom_event_id(iox2.EventId.new(PubSubEvent.ReceivedSample))
                break
        return result

    def subscribe(self, stream_name):
        try:
            self.subscriber_service = (
                self.node.service_builder(iox2.ServiceName.new(f"{stream_name}/COMPOSITE_BUFFER/Frame"))
                .publish_subscribe(iox2.Slice[ctypes.c_uint8])
                .user_header(ShmSerializedStreamHeader)
                .payload_alignment(iox2.Alignment.new(8))
                .history_size(1)
                .subscriber_max_buffer_size(4)
                .open()
            )

            self.subscriber = self.subscriber_service.subscriber_builder().create()
        except iox2.PublishSubscribeOpenError:
            log.error(f"error subscribing to channel for {stream_name}")
            return False
        return True

    def receive_frame(self, callback, cycle_time_ms=1):
        if not self.subscriber:
            log.error("missing subscriber")
            return False
        cycle_time = iox2.Duration.from_millis(cycle_time_ms)
        try:
            while True:
                sample = self.subscriber.receive()
                if sample is not None:
                    payload = sample.payload()

                    user_header = sample.user_header().contents
                    # print("received", payload.len(), "bytes")
                    # wrap data without copying for efficient decoding
                    buf_type = ctypes.c_uint8 * payload.len()
                    buf = ctypes.cast(payload.as_ptr(), ctypes.POINTER(buf_type)).contents
                    with decode_buffer_descriptor(memoryview(buf)) as message:
                        #log.debug(f"decoded frame: {user_header.timestamp}")
                        return callback(user_header, message)
                else:
                    self.node.wait(cycle_time)

        except iox2.NodeWaitFailure as e:
            log.exception(e)
        return False

    def teardown(self):
        if self.subscriber:
            self.subscriber = None
        if self.subscriber_service:
            self.subscriber_service = None