/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "shm_synchronized_buffer_receiver.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <type_traits>

#include <holoscan/logger/logger.hpp>

namespace tcn::shm {

// iceoryx2 type aliases
static constexpr iox2::ServiceType IpcServiceType = iox2::ServiceType::Ipc;
using SlicePayload = iox2::bb::Slice<uint8_t>;

// Regex for matching device context service names
static const std::regex kDeviceContextPattern(R"(^(.+)/DEVICE_CONTEXT/SensorCalibration$)");

// ---------------------------------------------------------------------------
// SubscriberState: holds iceoryx2 service + subscriber for frame data.
// All iceoryx2 objects are constructed in-place on the heap to avoid
// post-creation moves that may invalidate internal C FFI handles.
// ---------------------------------------------------------------------------
struct ShmSynchronizedBufferReceiver::SubscriberState {
    using PubSubService = std::remove_reference_t<decltype(
        std::declval<iox2::Node<IpcServiceType>>()
            .service_builder(std::declval<iox2::ServiceName>())
            .publish_subscribe<SlicePayload>()
            .user_header<ShmSerializedStreamHeader>()
            .payload_alignment(8)
            .history_size(1U)
            .subscriber_max_buffer_size(4U)
            .open()
            .value())>;

    using Subscriber = std::remove_reference_t<decltype(
        std::declval<PubSubService>()
            .subscriber_builder()
            .create()
            .value())>;

    PubSubService service;
    std::optional<Subscriber> subscriber;  // constructed after service is stable

    // Factory: first allocate service on the heap, THEN create subscriber
    // from the heap-resident service so the subscriber is never moved.
    static std::unique_ptr<SubscriberState> create(
        iox2::Node<IpcServiceType>& node,
        const std::string& service_name_str) {
        auto sname = iox2::ServiceName::create(service_name_str.c_str()).value();

        auto service = node.service_builder(sname)
            .publish_subscribe<SlicePayload>()
            .user_header<ShmSerializedStreamHeader>()
            .payload_alignment(8)
            .history_size(1U)
            .subscriber_max_buffer_size(4U)
            .open()
            .value();

        // Allocate on heap with service moved in; subscriber not yet created
        auto state = std::unique_ptr<SubscriberState>(
            new SubscriberState(std::move(service)));

        // Now create subscriber from the heap-resident service.
        // The subscriber is constructed directly in the optional — no moves.
        state->subscriber.emplace(
            state->service.subscriber_builder().create().value());

        return state;
    }

private:
    explicit SubscriberState(PubSubService&& svc)
        : service(std::move(svc)) {}
};

// ---------------------------------------------------------------------------
// Construction / destruction / move
// ---------------------------------------------------------------------------
ShmSynchronizedBufferReceiver::ShmSynchronizedBufferReceiver(
    iox2::Node<iox2::ServiceType::Ipc> node)
    : node_(std::move(node)) {}

ShmSynchronizedBufferReceiver::~ShmSynchronizedBufferReceiver() {
    teardown();
}

ShmSynchronizedBufferReceiver::ShmSynchronizedBufferReceiver(
    ShmSynchronizedBufferReceiver&&) noexcept = default;
ShmSynchronizedBufferReceiver& ShmSynchronizedBufferReceiver::operator=(
    ShmSynchronizedBufferReceiver&&) noexcept = default;

// ---------------------------------------------------------------------------
// discover_devices
// ---------------------------------------------------------------------------
std::vector<std::string> ShmSynchronizedBufferReceiver::discover_devices() {
    std::set<std::string> camera_names;

    auto result = iox2::Service<IpcServiceType>::list(
        iox2::Config::global_config(),
        [&camera_names](iox2::ServiceDetails<IpcServiceType> details) {
            std::string name_str(details.static_details.name());
            std::smatch match;
            if (std::regex_match(name_str, match, kDeviceContextPattern)) {
                camera_names.insert(match[1].str());
            }
            return iox2::CallbackProgression::Continue;
        });
    if (!result.has_value()) {
        HOLOSCAN_LOG_WARN("Failed to list iceoryx2 services for device discovery");
        return {};
    }

    return std::vector<std::string>(camera_names.begin(), camera_names.end());
}

// ---------------------------------------------------------------------------
// Helper: receive one ShmSerializedMessage, decode it, and apply a transform
// while the SHM sample is still alive (zero-copy friendly).
// ---------------------------------------------------------------------------
namespace {

/// Open a pub/sub service for ShmSerializedMessage, create subscriber + event
/// notifier, send SubscriberConnected event, block until one message arrives,
/// then decode and apply `transform` to the Cap'n Proto reader.  The iceoryx2
/// sample stays alive throughout the transform so the reader pointers are valid.
template <typename CapnpType, typename Transform>
auto receive_and_transform(
    iox2::Node<IpcServiceType>& node,
    const std::string& service_name_str,
    Transform&& transform)
    -> decltype(transform(std::declval<typename CapnpType::Reader>())) {

    auto sname = iox2::ServiceName::create(service_name_str.c_str()).value();

    // Open pub/sub service
    auto pubsub_service = node.service_builder(sname)
        .publish_subscribe<ShmSerializedMessage>()
        .history_size(1U)
        .subscriber_max_buffer_size(4U)
        .open()
        .value();

    // Open event service
    auto event_service = node.service_builder(sname)
        .event()
        .open_or_create()
        .value();

    auto subscriber = pubsub_service.subscriber_builder().create().value();
    auto notifier = event_service.notifier_builder().create().value();

    // Notify publisher that we connected
    notifier.notify_with_custom_event_id(
        iox2::EventId(static_cast<size_t>(PubSubEvent::SubscriberConnected)));

    // Block until we receive a sample
    while (true) {
        auto maybe_sample = subscriber.receive().value();
        if (maybe_sample.has_value()) {
            auto& sample = maybe_sample.value();
            auto& payload = sample.payload();
            auto data = payload.payload_data();
            auto size = payload.payload_len();

            HOLOSCAN_LOG_DEBUG("Received message from {} ({} bytes)", service_name_str, size);

            if (size == 0) {
                HOLOSCAN_LOG_ERROR("Empty payload from {} — skipping", service_name_str);
                continue;
            }
            if (size < 8) {
                HOLOSCAN_LOG_ERROR("Payload too small ({} bytes) from {} — need at least 8 for segment table",
                                   size, service_name_str);
                continue;
            }

            // Notify publisher we received the sample
            notifier.notify_with_custom_event_id(
                iox2::EventId(static_cast<size_t>(PubSubEvent::ReceivedSample)));

            // Decode and transform while the sample (and its SHM buffer) is alive.
            try {
                auto decoded = DecodedMessage<CapnpType>::decode(data, size);
                return transform(decoded.root());
            } catch (const std::exception& e) {
                HOLOSCAN_LOG_ERROR("Cap'n Proto decode/transform failed for {}: {}", service_name_str, e.what());
                throw;
            }
        }
        // Brief wait before retry
        node.wait(iox2::bb::Duration::from_millis(1));
    }
}

// Helper: convert Cap'n Proto CameraIntrinsicParameters to gxf::CameraModel.
nvidia::gxf::CameraModel camera_model_from_capnp(
    artekmed::shm::CameraIntrinsicParameters::Reader params) {
    nvidia::gxf::CameraModel model{};
    model.distortion_type = nvidia::gxf::DistortionType::Brown;
    model.dimensions.x = static_cast<uint32_t>(params.getWidth());
    model.dimensions.y = static_cast<uint32_t>(params.getHeight());
    model.focal_length.x = params.getFovX();
    model.focal_length.y = params.getFovY();
    model.principal_point.x = params.getCX();
    model.principal_point.y = params.getCY();
    model.skew_value = 1.0f;

    auto dist = params.getDistortionParams();
    model.distortion_coefficients[0] = dist.getK1();
    model.distortion_coefficients[1] = dist.getK2();
    model.distortion_coefficients[2] = dist.getTx();
    model.distortion_coefficients[3] = dist.getTy();
    model.distortion_coefficients[4] = dist.getK3();
    model.distortion_coefficients[5] = dist.getK4();
    model.distortion_coefficients[6] = dist.getK5();
    model.distortion_coefficients[7] = dist.getK6();

    return model;
}

// Helper: convert Cap'n Proto Pose to RigidTransform.
RigidTransform pose_from_capnp(artekmed::schema::Pose::Reader pose) {
    auto t = pose.getTranslation();
    auto r = pose.getRotation();
    return RigidTransform(
        Eigen::Vector3f(t.getX(), t.getY(), t.getZ()),
        Eigen::Quaternion<float>(r.getW(), r.getX(), r.getY(), r.getZ()));
}

}  // namespace

// ---------------------------------------------------------------------------
// retrieve_device_context — parse Cap'n Proto into native CameraDeviceInfo
// ---------------------------------------------------------------------------
CameraDeviceInfo ShmSynchronizedBufferReceiver::retrieve_device_context(
    const std::string& camera_name) {
    HOLOSCAN_LOG_INFO("Receive Camera Device Context: {}", camera_name);
    auto service_name = camera_name + "/DEVICE_CONTEXT/SensorCalibration";

    return receive_and_transform<artekmed::shm::ShmDeviceContext>(
        node_, service_name,
        [](artekmed::shm::ShmDeviceContext::Reader ctx) -> CameraDeviceInfo {
            CameraDeviceInfo info;
            info.depth_units_per_meter = ctx.getDepthUnitsPerMeter();
            info.is_valid = ctx.getIsValid();
            info.frame_rate = ctx.getFrameRate();

            if (ctx.hasCalibration()) {
                auto calib = ctx.getCalibration();
                info.depth_camera_model = camera_model_from_capnp(calib.getDepthCameraParameters());
                info.color_camera_model = camera_model_from_capnp(calib.getColorCameraParameters());
                info.camera_pose = pose_from_capnp(calib.getCameraPose());
                info.color_to_depth = pose_from_capnp(calib.getColor2depthTransform());
            }
            return info;
        });
}

// ---------------------------------------------------------------------------
// retrieve_channel_config — parse Cap'n Proto into native ChannelPortInfo
// ---------------------------------------------------------------------------
std::vector<ChannelPortInfo> ShmSynchronizedBufferReceiver::retrieve_channel_config(
    const std::string& stream_name) {
    auto service_name = stream_name + "/COMPOSITE_BUFFER/Config";
    HOLOSCAN_LOG_INFO("retrieve_channel_config({})", service_name);

    return receive_and_transform<artekmed::shm::ShmBufferConnectionStatus>(
        node_, service_name,
        [](artekmed::shm::ShmBufferConnectionStatus::Reader config) -> std::vector<ChannelPortInfo> {
            std::vector<ChannelPortInfo> ports;
            for (auto port : config.getPorts()) {
                ChannelPortInfo info;
                info.name = std::string(port.getName().cStr());
                auto status = port.getStatus();
                auto buffer_info = status.getBufferInfo();
                info.width = buffer_info.getWidth();
                info.height = buffer_info.getHeight();
                info.bits_per_element = buffer_info.getBitsPerElement();
                info.frame_size = buffer_info.getFrameSize();
                info.semantic_type = buffer_info.getSemanticType();

                auto pt = status.getPortType();
                if (pt == artekmed::schema::CameraPortType::DEPTHIMAGE) {
                    info.port_type = "depthimage";
                } else if (pt == artekmed::schema::CameraPortType::COLORIMAGE) {
                    info.port_type = "colorimage";
                }
                ports.push_back(std::move(info));
            }
            return ports;
        });
}

// ---------------------------------------------------------------------------
// subscribe
// ---------------------------------------------------------------------------
bool ShmSynchronizedBufferReceiver::subscribe(const std::string& stream_name) {
    auto service_name_str = stream_name + "/COMPOSITE_BUFFER/Frame";

    try {
        HOLOSCAN_LOG_INFO("Opening pub/sub service: {}", service_name_str);
        sub_state_ = SubscriberState::create(node_, service_name_str);
        HOLOSCAN_LOG_INFO("Subscribed to {} (sub_state_={:p}, subscriber={:p})",
                          service_name_str,
                          static_cast<void*>(sub_state_.get()),
                          static_cast<const void*>(&sub_state_->subscriber));

        // Notify the publisher that we connected so it calls update_connections()
        // and starts delivering frames to this subscriber.
        auto sname = iox2::ServiceName::create(service_name_str.c_str()).value();
        auto event_service = node_.service_builder(sname)
            .event()
            .open_or_create()
            .value();
        auto notifier = event_service.notifier_builder().create().value();
        notifier.notify_with_custom_event_id(
            iox2::EventId(static_cast<size_t>(PubSubEvent::SubscriberConnected)));
        HOLOSCAN_LOG_INFO("Sent SubscriberConnected event for {}", service_name_str);

        return true;
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Error subscribing to channel for {}: {}",
                           stream_name, e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// receive_frame (legacy callback API)
// ---------------------------------------------------------------------------
bool ShmSynchronizedBufferReceiver::receive_frame(
    const FrameCallback& callback, int cycle_time_ms) {
    if (!sub_state_ || !sub_state_->subscriber.has_value()) {
        HOLOSCAN_LOG_ERROR("Missing subscriber (sub_state_ or subscriber is null)");
        return false;
    }

    auto cycle_time = iox2::bb::Duration::from_millis(static_cast<uint64_t>(cycle_time_ms));

    try {
        while (true) {
            auto receive_result = sub_state_->subscriber->receive();
            if (!receive_result.has_value()) {
                auto err = receive_result.error();
                auto err_str = iox2::bb::from<iox2::ReceiveError, const char*>(err);
                if (err == iox2::ReceiveError::FailedToEstablishConnection) {
                    HOLOSCAN_LOG_DEBUG("receive(): {} — waiting for publisher", err_str);
                } else {
                    HOLOSCAN_LOG_WARN("receive(): {} — retrying", err_str);
                }
                node_.wait(cycle_time);
                continue;
            }
            auto& maybe_sample = receive_result.value();
            if (maybe_sample.has_value()) {
                auto& sample = maybe_sample.value();

                // Access user header (stream timestamp etc.)
                auto& user_header = sample.user_header();

                // Access payload (raw frame bytes)
                auto payload_slice = sample.payload();
                auto* data = payload_slice.data();
                auto size = payload_slice.number_of_bytes();

                // Decode the buffer descriptor from the frame payload
                auto decoded = DecodedBufferDescriptor::decode(data, size);
                return callback(user_header, decoded.root());
            } else {
                node_.wait(cycle_time);
            }
        }
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Error receiving frame: {}", e.what());
    }
    return false;
}

// ---------------------------------------------------------------------------
// receive_frame_zero_copy — returns a frame with raw pointers into SHM.
// The iceoryx2 sample is kept alive via a shared_ptr so that callers can
// perform async CUDA copies before releasing the segment.
// ---------------------------------------------------------------------------
std::optional<ShmZeroCopyFrame> ShmSynchronizedBufferReceiver::receive_frame_zero_copy(
    int cycle_time_ms) {
    if (!sub_state_ || !sub_state_->subscriber.has_value()) {
        HOLOSCAN_LOG_ERROR("Missing subscriber (sub_state_ or subscriber is null)");
        return std::nullopt;
    }

    try {
        auto receive_result = sub_state_->subscriber->receive();
        if (!receive_result.has_value()) {
            auto err = receive_result.error();
            auto err_str = iox2::bb::from<iox2::ReceiveError, const char*>(err);
            if (err == iox2::ReceiveError::FailedToEstablishConnection) {
                HOLOSCAN_LOG_DEBUG("receive(): {} — waiting for publisher", err_str);
            } else {
                HOLOSCAN_LOG_WARN("receive(): {} — retrying", err_str);
            }
            node_.wait(iox2::bb::Duration::from_millis(
                static_cast<uint64_t>(cycle_time_ms)));
            return std::nullopt;
        }
        auto& maybe_sample = receive_result.value();
        if (!maybe_sample.has_value()) {
            // No data yet — brief wait, then return to let caller check stop flag
            node_.wait(iox2::bb::Duration::from_millis(
                static_cast<uint64_t>(cycle_time_ms)));
            return std::nullopt;
        }

        auto& sample = maybe_sample.value();
        auto& user_header = sample.user_header();
        auto payload_slice = sample.payload();
        auto* data = payload_slice.data();
        auto size = payload_slice.number_of_bytes();

        // Decode capnp in-place — reader pointers go directly into SHM
        auto decoded = DecodedBufferDescriptor::decode(data, size);
        auto descriptor = decoded.root();

        ShmZeroCopyFrame frame;
        frame.timestamp = user_header.timestamp;

        for (auto port : descriptor.getPorts()) {
            auto port_data = port.getData();
            auto port_type = port_data.getPortType();

            // Only process color and depth image ports
            if (port_type != artekmed::schema::CameraPortType::COLORIMAGE &&
                port_type != artekmed::schema::CameraPortType::DEPTHIMAGE) {
                continue;
            }

            ShmPortView pv;
            pv.name = std::string(port.getName().cStr());
            pv.is_color = (port_type == artekmed::schema::CameraPortType::COLORIMAGE);

            auto metadata = port_data.getMetadata();
            auto stream_header = metadata.getHeader();
            pv.width = stream_header.getDimX();
            pv.height = stream_header.getDimY();
            int32_t bits_per_element = stream_header.getBitsPerElement();
            pv.channels = pv.is_color ? (bits_per_element / 8) : 1;

            auto raw_data = port_data.getData();
            pv.data_ptr = reinterpret_cast<const uint8_t*>(raw_data.begin());
            pv.data_size = raw_data.size();

            frame.ports.push_back(std::move(pv));
        }

        if (frame.ports.empty()) {
            return std::nullopt;
        }

        // Move sample + decoded reader to the heap so the SHM segment stays
        // alive until the caller drops the frame.  We define the lifetime
        // holder locally because the iceoryx2 sample type is complex and
        // involves private SubscriberState types.
        using SampleOptType = std::decay_t<decltype(maybe_sample)>;
        struct ShmLifetime {
            SampleOptType sample;
            DecodedBufferDescriptor decoded;
        };
        frame.shm_handle = std::make_shared<ShmLifetime>(
            ShmLifetime{std::move(maybe_sample), std::move(decoded)});

        return frame;

    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Error in zero-copy frame receive: {}", e.what());
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// teardown
// ---------------------------------------------------------------------------
void ShmSynchronizedBufferReceiver::teardown() {
    sub_state_.reset();
}

}  // namespace tcn::shm
