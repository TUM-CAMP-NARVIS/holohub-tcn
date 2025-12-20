import iceoryx2 as iox2
from shm_types import ShmSerializedMessage, PubSubEvent
from shm_serde import decode_shm_buffer_connection_status, decode_shm_device_context

class ShmSynchronizedBufferReceiver:

    def __init__(self, node):
        self.node = node

    def retrieve_channel_config(self, stream_name):
        result = None
        service = (
            self.node.service_builder(iox2.ServiceName.new(f"{stream_name}/COMPOSITE_BUFFER/Config"))
            .publish_subscribe(ShmSerializedMessage)
            .history_size(1)
            .subscriber_max_buffer_size(4)
            .open()
        )
        event = (
            self.node.service_builder(
                iox2.ServiceName.new(f"{stream_name}/COMPOSITE_BUFFER/Config")).event().open_or_create()
        )
        subscriber = service.subscriber_builder().create()
        notifier = event.notifier_builder().create()

        notifier.notify_with_custom_event_id(iox2.EventId.new(PubSubEvent.SubscriberConnected))

        while True:
            sample = subscriber.receive()
            if sample is not None:
                contents = sample.payload().contents
                print("received config payload", contents)
                with decode_shm_buffer_connection_status(contents.payload_bytes()) as message:
                    result = message.to_dict()
                    print("decoded config message", result)
                notifier.notify_with_custom_event_id(iox2.EventId.new(PubSubEvent.ReceivedSample))
                break
        return result

    def retrieve_device_context(self, camera_name):
        print("Receive Camera Device Context: ", camera_name)
        result = None
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

        message = None
        while True:
            sample = subscriber.receive()
            if sample is not None:
                contents = sample.payload().contents
                print("received device_context payload", contents)
                with decode_shm_device_context(contents.payload_bytes()) as message:
                    result = message.to_dict()
                    print("decoded device_context message", result)
                notifier.notify_with_custom_event_id(iox2.EventId.new(PubSubEvent.ReceivedSample))
                break

        return result
