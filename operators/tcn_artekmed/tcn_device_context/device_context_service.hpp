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

#include <map>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <gxf/multimedia/camera.hpp>

#include <holoscan/core/fragment_service.hpp>

#include "../common/datatypes.hpp"
#include "../tcn_shm_subscriber/shm_synchronized_buffer_receiver.hpp"

#include <xylt/library.h>

namespace tcn::ops {

/**
 * @brief Service for camera calibration data and XY lookup table generation.
 *
 * C++ port of Python DeviceContextService. Provides:
 * - Camera model extraction from Cap'n Proto device contexts
 * - Depth/color camera intrinsics as gxf::CameraModel
 * - Extrinsic transforms (camera pose, color-to-depth)
 * - XY lookup table generation via xylt library
 *
 * Unlike the Python version (which extends DefaultFragmentService),
 * this is a plain C++ class since Holoscan C++ SDK does not expose
 * a fragment service base class.
 */
class DeviceContextService : public holoscan::DefaultFragmentService {
public:
    DeviceContextService() = default;

    /// Add a device context for a camera (native struct, already parsed from Cap'n Proto).
    void add_device_context(const std::string& camera_name,
                            tcn::shm::CameraDeviceInfo device_info);

    /// Extract camera name from a port name like "camera0_depth".
    std::optional<std::string> get_camera_name_from_port_name(
        const std::string& port_name) const;

    /// Check if a camera is registered.
    bool has_camera(const std::string& camera_name) const;

    /// Get list of registered camera names.
    std::vector<std::string> camera_names() const;

    /// Get depth camera model for a given camera.
    std::optional<nvidia::gxf::CameraModel> get_depth_camera_model(
        const std::string& camera_name) const;

    /// Get color camera model for a given camera.
    std::optional<nvidia::gxf::CameraModel> get_color_camera_model(
        const std::string& camera_name) const;

    /// Get XY lookup table intrinsics for a given camera.
    std::optional<IntrinsicParameters> get_xy_table_intrinsics(
        const std::string& camera_name) const;

    /// Generate XY lookup table for depth reprojection.
    /// Returns table data as (height * width * 2) float vector, or nullopt on failure.
    std::shared_ptr<XYTableData> get_xy_table(const std::string& camera_name) const;

    /// Get depth camera extrinsics (camera pose).
    std::optional<RigidTransform> get_depth_extrinsics(
        const std::string& camera_name) const;

    /// Get color-to-depth transform.
    std::optional<RigidTransform> get_color_to_depth(
        const std::string& camera_name) const;

    /// Get inverse color-to-depth transform.
    std::optional<RigidTransform> get_color_to_depth_inv(
        const std::string& camera_name) const;

private:
    std::map<std::string, tcn::shm::CameraDeviceInfo> device_infos_;
    std::regex portname_pattern_{R"(^(camera[0-9]+)_.*$)"};
};

}  // namespace tcn::ops
