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

#include "shm_sender_op.hpp"

#include <chrono>
#include <cstring>
#include <optional>
#include <type_traits>

#include <holoscan/utils/cuda_macros.hpp>

#include <capnp/message.h>
#include <capnp/serialize.h>

#include "../tcn_shm_serde/generated/shm_synchronized_transport.capnp.h"
#include "../tcn_shm_serde/generated/enumerations.capnp.h"

namespace tcn::ops {

// iceoryx2 type aliases (same as subscriber)
static constexpr iox2::ServiceType IpcServiceType = iox2::ServiceType::Ipc;
using SlicePayload = iox2::bb::Slice<uint8_t>;

// ---------------------------------------------------------------------------
// PublisherState: holds iceoryx2 service + publisher for frame data.
// Allocated on the heap to avoid post-creation moves that invalidate
// internal C FFI handles (same pattern as subscriber).
// ---------------------------------------------------------------------------
struct TcnShmZenohSenderOp::PublisherState {
    using PubSubService = std::remove_reference_t<decltype(
        std::declval<iox2::Node<IpcServiceType>>()
            .service_builder(std::declval<iox2::ServiceName>())
            .publish_subscribe<SlicePayload>()
            .user_header<tcn::shm::ShmSerializedStreamHeader>()
            .payload_alignment(8)
            .history_size(1U)
            .subscriber_max_buffer_size(4U)
            .open_or_create()
            .value())>;

    using Publisher = std::remove_reference_t<decltype(
        std::declval<PubSubService>()
            .publisher_builder()
            .initial_max_slice_len(16 * 1024 * 1024)
            .allocation_strategy(iox2::AllocationStrategy::PowerOfTwo)
            .create()
            .value())>;

    PubSubService service;
    std::optional<Publisher> publisher;

    static std::unique_ptr<PublisherState> create(
        iox2::Node<IpcServiceType>& node,
        const std::string& service_name_str,
        size_t initial_slice_len) {
        auto sname = iox2::ServiceName::create(service_name_str.c_str()).value();

        auto service = node.service_builder(sname)
            .publish_subscribe<SlicePayload>()
            .user_header<tcn::shm::ShmSerializedStreamHeader>()
            .payload_alignment(8)
            .history_size(1U)
            .subscriber_max_buffer_size(4U)
            .open_or_create()
            .value();

        auto state = std::unique_ptr<PublisherState>(
            new PublisherState(std::move(service)));

        state->publisher.emplace(
            state->service.publisher_builder()
                .initial_max_slice_len(initial_slice_len)
                .allocation_strategy(iox2::AllocationStrategy::PowerOfTwo)
                .create()
                .value());

        return state;
    }

private:
    explicit PublisherState(PubSubService&& svc)
        : service(std::move(svc)) {}
};

// Out-of-line destructor (PublisherState is complete here)
TcnShmZenohSenderOp::~TcnShmZenohSenderOp() = default;

// ---------------------------------------------------------------------------
// Operator lifecycle
// ---------------------------------------------------------------------------

void TcnShmZenohSenderOp::setup(holoscan::OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("frame_input");

    spec.param(stream_name_, "stream_name",
               "Stream Name",
               "SHM service name prefix for publishing",
               std::string{"camera_streams"});
    spec.param(input_tensor_names_, "input_tensor_names",
               "Input Tensor Names",
               "List of tensor names to publish from the input entity",
               std::vector<std::string>{});
}

void TcnShmZenohSenderOp::initialize() {
    Operator::initialize();
}

void TcnShmZenohSenderOp::start() {
    // Create iceoryx2 node
    auto node_result = iox2::NodeBuilder()
        .name(iox2::NodeName::create("tcn_shm_sender").value())
        .create<IpcServiceType>();
    if (!node_result.has_value()) {
        HOLOSCAN_LOG_ERROR("Failed to create iceoryx2 node");
        return;
    }
    node_ = std::make_unique<iox2::Node<IpcServiceType>>(std::move(node_result.value()));

    // Create publisher service: {stream_name}/COMPOSITE_BUFFER/Frame
    auto service_name = stream_name_.get() + "/COMPOSITE_BUFFER/Frame";
    constexpr size_t kInitialSliceLen = 1024 * 1024;  // 1 MB initial, grows via PowerOfTwo

    try {
        pub_state_ = PublisherState::create(*node_, service_name, kInitialSliceLen);
        HOLOSCAN_LOG_INFO("SHM publisher created for service: {}", service_name);
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Failed to create SHM publisher for {}: {}",
                           service_name, e.what());
        return;
    }

    // Create CUDA stream for GPU->CPU copies
    HOLOSCAN_CUDA_CALL(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));

    frames_published_ = 0;
    HOLOSCAN_LOG_INFO("TcnShmZenohSenderOp started for stream: {}", stream_name_.get());
}

void TcnShmZenohSenderOp::stop() {
    pub_state_.reset();
    node_.reset();

    if (copy_stream_) {
        cudaStreamSynchronize(copy_stream_);
        cudaStreamDestroy(copy_stream_);
        copy_stream_ = nullptr;
    }

    staging_buffer_.clear();
    HOLOSCAN_LOG_INFO("TcnShmZenohSenderOp stopped ({} frames published)", frames_published_);
}

void TcnShmZenohSenderOp::compute(
    holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context) {

    if (!pub_state_ || !pub_state_->publisher.has_value()) {
        HOLOSCAN_LOG_ERROR("SHM publisher not initialized");
        return;
    }

    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("frame_input");
    if (!maybe_entity) {
        HOLOSCAN_LOG_DEBUG("No input entity received");
        return;
    }
    auto entity = maybe_entity.value();

    // Collect tensors to publish
    auto tensor_names = input_tensor_names_.get();
    if (tensor_names.empty()) {
        HOLOSCAN_LOG_WARN("No tensor names configured — skipping frame");
        return;
    }

    // Gather tensor info for all ports
    struct PortInfo {
        std::string name;
        nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor;
        int32_t width;
        int32_t height;
        int32_t channels;
        int32_t bits_per_element;
        size_t data_size;
        bool is_color;
        artekmed::schema::CameraPortType port_type;
        artekmed::schema::PixelFormat pixel_format;
    };
    std::vector<PortInfo> ports;

    for (const auto& name : tensor_names) {
        auto maybe_tensor = entity.get<nvidia::gxf::Tensor>(name.c_str());
        if (!maybe_tensor) {
            HOLOSCAN_LOG_DEBUG("Tensor '{}' not found in entity — skipping", name);
            continue;
        }
        auto tensor = maybe_tensor.value();
        auto shape = tensor->shape();
        if (shape.rank() < 2) {
            HOLOSCAN_LOG_WARN("Tensor '{}' has rank {} < 2 — skipping", name, shape.rank());
            continue;
        }

        PortInfo pi;
        pi.name = name;
        pi.tensor = tensor;
        pi.height = shape.dimension(0);
        pi.width = shape.dimension(1);
        pi.channels = (shape.rank() >= 3) ? shape.dimension(2) : 1;
        auto element_size = nvidia::gxf::PrimitiveTypeSize(tensor->element_type());
        pi.bits_per_element = static_cast<int32_t>(element_size * 8 * pi.channels);
        pi.data_size = static_cast<size_t>(pi.height) * pi.width * pi.channels * element_size;

        // Determine port type and pixel format from channel count and element type
        if (pi.channels >= 3) {
            pi.is_color = true;
            pi.port_type = artekmed::schema::CameraPortType::COLORIMAGE;
            pi.pixel_format = (pi.channels == 4)
                ? artekmed::schema::PixelFormat::RGBA
                : artekmed::schema::PixelFormat::RGB;
        } else {
            pi.is_color = false;
            pi.port_type = artekmed::schema::CameraPortType::DEPTHIMAGE;
            pi.pixel_format = artekmed::schema::PixelFormat::DEPTH;
        }

        ports.push_back(std::move(pi));
    }

    if (ports.empty()) {
        return;
    }

    // Compute total frame data size across all ports
    size_t total_frame_bytes = 0;
    for (const auto& p : ports) {
        total_frame_bytes += p.data_size;
    }

    // Build Cap'n Proto message describing the frame
    capnp::MallocMessageBuilder message_builder;
    auto descriptor = message_builder.initRoot<artekmed::shm::ShmBufferDescriptor>();

    auto now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    descriptor.setTimestamp(now_ns);
    descriptor.setNumPorts(static_cast<int16_t>(ports.size()));
    auto capnp_ports = descriptor.initPorts(static_cast<unsigned int>(ports.size()));

    // Track where each port's data starts in the serialized payload
    // The Cap'n Proto message will contain inline Data blobs for each port
    for (size_t i = 0; i < ports.size(); ++i) {
        auto& pi = ports[i];
        auto port_entry = capnp_ports[i];
        port_entry.setName(pi.name.c_str());

        auto port_data = port_entry.initData();
        port_data.setPortType(pi.port_type);

        auto metadata = port_data.initMetadata();
        metadata.setTimestamp(now_ns);
        auto header = metadata.initHeader();
        header.setDimX(pi.width);
        header.setDimY(pi.height);
        header.setBitsPerElement(pi.bits_per_element);
        header.setBufferLen(static_cast<uint64_t>(pi.data_size));
        header.setTimestamp(now_ns);
        header.setImage(pi.pixel_format);

        // Pre-allocate the data blob in the Cap'n Proto message
        auto data_blob = port_data.initData(static_cast<unsigned int>(pi.data_size));

        // Copy frame data into the Cap'n Proto blob
        // First, get the data from GPU if needed
        auto maybe_data = pi.tensor->data<uint8_t>();
        if (!maybe_data) {
            HOLOSCAN_LOG_ERROR("Failed to get tensor data pointer for '{}'", pi.name);
            continue;
        }

        auto* src_ptr = maybe_data.value();
        auto storage_type = pi.tensor->storage_type();

        if (storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
            // GPU memory — need to copy to CPU first
            if (staging_buffer_.size() < pi.data_size) {
                staging_buffer_.resize(pi.data_size);
            }
            HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
                staging_buffer_.data(), src_ptr, pi.data_size,
                cudaMemcpyDeviceToHost, copy_stream_));
            HOLOSCAN_CUDA_CALL(cudaStreamSynchronize(copy_stream_));
            std::memcpy(data_blob.begin(), staging_buffer_.data(), pi.data_size);
        } else {
            // Host memory — direct copy
            std::memcpy(data_blob.begin(), src_ptr, pi.data_size);
        }
    }

    // Serialize the Cap'n Proto message to a flat array
    auto serialized = capnp::messageToFlatArray(message_builder);
    auto serialized_bytes = serialized.asBytes();

    // Loan SHM sample and publish
    auto& publisher = pub_state_->publisher.value();
    auto loan_result = publisher.loan_slice(serialized_bytes.size());
    if (!loan_result.has_value()) {
        HOLOSCAN_LOG_WARN("SHM loan failed (back-pressure?) — dropping frame");
        return;
    }
    auto sample = std::move(loan_result.value());

    // Fill user header (explicit template required by iceoryx2 API)
    auto& user_header =
        sample.template user_header_mut<tcn::shm::ShmSerializedStreamHeader>();
    user_header.timestamp = now_ns;

    // Copy serialized data into SHM slice (explicit template required)
    auto payload = sample.template payload_mut<iox2::bb::Slice<uint8_t>>();
    std::memcpy(payload.data(), serialized_bytes.begin(), serialized_bytes.size());

    // Publish via iceoryx2 global send
    ::iox2::send(std::move(sample)).value();

    frames_published_++;
    if (frames_published_ % 300 == 0) {
        HOLOSCAN_LOG_INFO("SHM sender: {} frames published", frames_published_);
    }
}

}  // namespace tcn::ops
