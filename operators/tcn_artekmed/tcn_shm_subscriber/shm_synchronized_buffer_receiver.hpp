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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gxf/multimedia/camera.hpp>

#include "iox2/iceoryx2.hpp"

#include "../common/datatypes.hpp"
#include "../tcn_shm_serde/shm_types.hpp"
#include "../tcn_shm_serde/shm_serde.hpp"

namespace tcn::shm {

// ---------------------------------------------------------------------------
// Native metadata structs — populated from Cap'n Proto while the SHM sample
// is still alive, then the sample is released.  No buffer copies needed.
// ---------------------------------------------------------------------------

/// Per-camera calibration and configuration extracted from ShmDeviceContext.
struct CameraDeviceInfo {
    nvidia::gxf::CameraModel depth_camera_model{};
    nvidia::gxf::CameraModel color_camera_model{};
    RigidTransform camera_pose;
    RigidTransform color_to_depth;
    float depth_units_per_meter = 0.0f;
    bool is_valid = false;
    int32_t frame_rate = 0;
};

/// Per-port channel info extracted from ShmBufferConnectionStatus.
struct ChannelPortInfo {
    std::string name;
    std::string port_type;        // "depthimage" or "colorimage"
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t bits_per_element = 0;
    uint64_t frame_size = 0;
};

/// Callback type for receive_frame: receives the stream header and decoded buffer descriptor.
/// Returns true to continue receiving, false to stop.
using FrameCallback = std::function<bool(
    const ShmSerializedStreamHeader& header,
    artekmed::shm::ShmBufferDescriptor::Reader descriptor)>;

/**
 * @brief Low-level iceoryx2 subscriber for receiving SHM camera data.
 *
 * C++ port of Python ShmSynchronizedBufferReceiver. Provides:
 * - Camera device discovery via iceoryx2 service listing
 * - Device context (calibration) retrieval → native CameraDeviceInfo
 * - Channel configuration retrieval → native ChannelPortInfo vector
 * - Blocking frame reception with zero-copy buffer access
 *
 * Owns the iceoryx2 Node so that all services created from it remain valid.
 */
class ShmSynchronizedBufferReceiver {
public:
    static constexpr iox2::ServiceType IpcService = iox2::ServiceType::Ipc;

    explicit ShmSynchronizedBufferReceiver(iox2::Node<iox2::ServiceType::Ipc> node);
    ~ShmSynchronizedBufferReceiver();

    ShmSynchronizedBufferReceiver(const ShmSynchronizedBufferReceiver&) = delete;
    ShmSynchronizedBufferReceiver& operator=(const ShmSynchronizedBufferReceiver&) = delete;
    ShmSynchronizedBufferReceiver(ShmSynchronizedBufferReceiver&&) noexcept;
    ShmSynchronizedBufferReceiver& operator=(ShmSynchronizedBufferReceiver&&) noexcept;

    /// Discover available camera devices by listing iceoryx2 services matching
    /// the pattern "{camera_name}/DEVICE_CONTEXT/SensorCalibration".
    static std::vector<std::string> discover_devices();

    /// Retrieve device context (calibration data) for a specific camera.
    /// Parses the Cap'n Proto message into native structs while the SHM
    /// sample is alive, then releases the sample.
    CameraDeviceInfo retrieve_device_context(const std::string& camera_name);

    /// Retrieve channel configuration for a given stream.
    /// Returns native ChannelPortInfo structs parsed while the SHM sample
    /// is alive.
    std::vector<ChannelPortInfo> retrieve_channel_config(const std::string& stream_name);

    /// Subscribe to the frame data stream for a given stream name.
    /// Returns true on success, false on error.
    bool subscribe(const std::string& stream_name);

    /// Blocking frame receive. Calls callback with each received frame.
    /// Blocks until a frame is received or an error occurs.
    /// @param callback Called with (header, descriptor) for each frame.
    /// @param cycle_time_ms Wait time between polls in milliseconds.
    /// @return true if callback returned true, false on error or callback returned false.
    bool receive_frame(const FrameCallback& callback, int cycle_time_ms = 1);

    /// Clean up all subscribers and services.
    void teardown();

private:
    // Node MUST be declared before sub_state_ so that subscribers are
    // destroyed before the node they were created from.
    iox2::Node<iox2::ServiceType::Ipc> node_;

    // Frame data subscriber (Slice<uint8_t> payload with ShmSerializedStreamHeader user header)
    struct SubscriberState;
    std::unique_ptr<SubscriberState> sub_state_;
};

}  // namespace tcn::shm
