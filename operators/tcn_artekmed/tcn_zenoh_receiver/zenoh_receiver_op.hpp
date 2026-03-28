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

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <cuda_runtime.h>

// Forward-declare zenoh types to avoid pulling the full header into every consumer.
namespace zenoh {
class Session;
template <class Handler>
class Subscriber;
}  // namespace zenoh

namespace tcn::ops {

/// Configuration for a single discovered Zenoh video stream.
struct ZenohStreamConfig {
    std::string name;          ///< Unique stream name (e.g. "camera_0_color")
    std::string topic;         ///< Zenoh key expression for data subscription
    std::string sensor_name;   ///< Source sensor (e.g. "camera_0")
    int32_t stream_index = 0;  ///< Zero-based index
    // Descriptor metadata (populated during discovery)
    int32_t image_width = 0;
    int32_t image_height = 0;
    int32_t image_step = 0;
    int32_t image_format = 0;       ///< Pixel format enum
    int32_t image_compression = 0;  ///< Compression enum (0=raw, 1=H264, 2=H265)
    float frame_rate = 0.0f;
};

/// A received Zenoh sample with CDR type and raw bytes.
struct ReceivedSample {
    std::string type_name;
    std::vector<uint8_t> payload;
};

/**
 * @brief Composite Holoscan source operator: Zenoh subscription + CDR decode + GPU output.
 *
 * Subscribes to multiple Zenoh video streams, CDR-decodes payloads using the
 * CdrTypeRegistry, and emits decoded frames as GPU tensors on dynamic output ports.
 *
 * The operator does NOT perform Zenoh discovery itself. Instead, the application
 * calls the static `discover_streams()` helper (or builds ZenohStreamConfig
 * manually), then passes configs via `set_stream_configs()` before setup().
 *
 * This follows the same two-phase pattern as TcnStreamSplitterOp:
 *   1. App creates operator, calls set_stream_configs() + set_session()
 *   2. init_spec() finalizes the OperatorSpec with dynamic output ports
 *   3. start() subscribes to Zenoh topics
 *   4. compute() dequeues, decodes, uploads to GPU, emits
 *
 * Output ports:
 *   - One per discovered stream, named by ZenohStreamConfig::name
 *   - Each emits holoscan::gxf::Entity containing a GPU tensor ("" component)
 *   - type_name_{stream}: CDR type name string (for each stream)
 *
 * Parameters:
 *   - async_condition: AsynchronousCondition for event-driven scheduling
 *   - allocator: GPU memory allocator (required: UnboundedAllocator or BlockMemoryPool)
 *   - cuda_stream_pool: CudaStreamPool for async GPU operations
 */
class TcnZenohReceiverOp : public holoscan::Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnZenohReceiverOp)

    TcnZenohReceiverOp() = default;

    /// Set discovered stream configs (must be called before setup/init_spec).
    void set_stream_configs(std::vector<ZenohStreamConfig> configs) {
        stream_configs_ = std::move(configs);
    }

    /// Set the Zenoh session (must be called before start).
    void set_session(std::shared_ptr<zenoh::Session> session) {
        session_ = std::move(session);
    }

    /// Create OperatorSpec after stream configs are set (manual construction path).
    void init_spec() {
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment());
        setup(*spec_);
    }

    /// Return the current stream configs (for application-level wiring).
    const std::vector<ZenohStreamConfig>& stream_configs() const {
        return stream_configs_;
    }

    // --- Static discovery helpers ---

    /// Discover camera sensors and resolve stream descriptors via Zenoh RPC.
    /// Returns stream configs ready to pass to set_stream_configs().
    ///
    /// Protocol:
    ///   1. GET {topic_prefix}/{capture_node}/rpc/sensor/*/describe
    ///   2. For each sensor with color/depth enabled:
    ///      GET {topic_prefix}/{sensor_name}/cfg/dsc/{color|depth}_image_bitstream
    ///   3. Parse StreamDescriptorMessage for stream topic and metadata
    static std::vector<ZenohStreamConfig> discover_streams(
        zenoh::Session& session,
        const std::string& topic_prefix,
        const std::string& capture_node,
        const std::vector<std::string>& stream_types);

    void setup(holoscan::OperatorSpec& spec) override;
    void initialize() override;
    void start() override;
    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override;
    void stop() override;

 private:
    /// Per-stream subscription state.
    struct StreamState {
        ZenohStreamConfig config;
        std::unique_ptr<zenoh::Subscriber<void>> subscriber;
        std::mutex queue_mutex;
        std::queue<ReceivedSample> sample_queue;
        std::atomic<uint64_t> samples_dropped{0};
        std::atomic<uint64_t> frames_emitted{0};

        static constexpr size_t kMaxQueuedSamples = 4;
    };

    std::vector<ZenohStreamConfig> stream_configs_;
    std::vector<std::unique_ptr<StreamState>> stream_states_;
    std::shared_ptr<zenoh::Session> session_;

    // Parameters
    holoscan::Parameter<std::shared_ptr<holoscan::AsynchronousCondition>> async_condition_;
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    holoscan::Parameter<std::shared_ptr<holoscan::CudaStreamPool>> cuda_stream_pool_;

    // CUDA stream for host-to-device copies
    cudaStream_t upload_stream_ = nullptr;
};

}  // namespace tcn::ops
