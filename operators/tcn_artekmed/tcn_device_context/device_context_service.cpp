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

#include "device_context_service.hpp"

#include <holoscan/logger/logger.hpp>

namespace tcn::ops {

void DeviceContextService::add_device_context(
    const std::string& camera_name,
    tcn::shm::CameraDeviceInfo device_info) {
    device_infos_.emplace(camera_name, std::move(device_info));
}

std::optional<std::string> DeviceContextService::get_camera_name_from_port_name(
    const std::string& port_name) const {
    std::smatch match;
    if (std::regex_match(port_name, match, portname_pattern_)) {
        return match[1].str();
    }
    return std::nullopt;
}

bool DeviceContextService::has_camera(const std::string& camera_name) const {
    return device_infos_.count(camera_name) > 0;
}

std::vector<std::string> DeviceContextService::camera_names() const {
    std::vector<std::string> names;
    names.reserve(device_infos_.size());
    for (const auto& [name, _] : device_infos_) {
        names.push_back(name);
    }
    return names;
}

std::optional<nvidia::gxf::CameraModel> DeviceContextService::get_depth_camera_model(
    const std::string& camera_name) const {
    auto it = device_infos_.find(camera_name);
    if (it == device_infos_.end()) {
        HOLOSCAN_LOG_ERROR("No camera found with name: {}", camera_name);
        return std::nullopt;
    }
    return it->second.depth_camera_model;
}

std::optional<nvidia::gxf::CameraModel> DeviceContextService::get_color_camera_model(
    const std::string& camera_name) const {
    auto it = device_infos_.find(camera_name);
    if (it == device_infos_.end()) {
        HOLOSCAN_LOG_ERROR("No camera found with name: {}", camera_name);
        return std::nullopt;
    }
    return it->second.color_camera_model;
}

std::optional<IntrinsicParameters> DeviceContextService::get_xy_table_intrinsics(
    const std::string& camera_name) const {
    auto model = get_depth_camera_model(camera_name);
    if (!model) {
        return std::nullopt;
    }

    IntrinsicParameters intrinsics;
    intrinsics.fov_x = model->focal_length.x;
    intrinsics.fov_y = model->focal_length.y;
    intrinsics.c_x = model->principal_point.x;
    intrinsics.c_y = model->principal_point.y;
    intrinsics.width = model->dimensions.x;
    intrinsics.height = model->dimensions.y;
    intrinsics.tangential_distortion = {
        model->distortion_coefficients[2],  // tx
        model->distortion_coefficients[3],  // ty
    };
    intrinsics.radial_distortion = {
        model->distortion_coefficients[0],  // k1
        model->distortion_coefficients[1],  // k2
        model->distortion_coefficients[4],  // k3
        model->distortion_coefficients[5],  // k4
        model->distortion_coefficients[6],  // k5
        model->distortion_coefficients[7],  // k6
    };
    return intrinsics;
}

std::shared_ptr<XYTableData> DeviceContextService::get_xy_table(
    const std::string& camera_name) const {
    auto intrinsics = get_xy_table_intrinsics(camera_name);
    if (!intrinsics) {
        return nullptr;
    }

    HOLOSCAN_LOG_INFO("Creating xy-table for {}", camera_name);
    auto xy_table = create_xy_lookup_table_from_intrinsics(*intrinsics);

    if (!xy_table) {
        HOLOSCAN_LOG_ERROR("Failed to create XY lookup table for camera {}", camera_name);
        return nullptr;
    }
    if (xy_table->width == 0 || xy_table->height == 0 || xy_table->data.empty()) {
        HOLOSCAN_LOG_ERROR("XY lookup table for camera {} is empty", camera_name);
        return nullptr;
    }

    return xy_table;
}

std::optional<RigidTransform> DeviceContextService::get_depth_extrinsics(
    const std::string& camera_name) const {
    auto it = device_infos_.find(camera_name);
    if (it == device_infos_.end()) {
        HOLOSCAN_LOG_ERROR("No camera found with name: {}", camera_name);
        return std::nullopt;
    }
    return it->second.camera_pose;
}

std::optional<RigidTransform> DeviceContextService::get_color_to_depth(
    const std::string& camera_name) const {
    auto it = device_infos_.find(camera_name);
    if (it == device_infos_.end()) {
        HOLOSCAN_LOG_ERROR("No camera found with name: {}", camera_name);
        return std::nullopt;
    }
    return it->second.color_to_depth;
}

std::optional<RigidTransform> DeviceContextService::get_color_to_depth_inv(
    const std::string& camera_name) const {
    auto it = device_infos_.find(camera_name);
    if (it == device_infos_.end()) {
        HOLOSCAN_LOG_ERROR("No camera found with name: {}", camera_name);
        return std::nullopt;
    }
    return it->second.color_to_depth.inverse();
}

}  // namespace tcn::ops
