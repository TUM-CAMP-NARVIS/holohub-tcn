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

namespace doc {

namespace DeviceContextService {
constexpr const char* doc_DeviceContextService = R"doc(
Service for camera calibration data and XY lookup table generation.

Wraps Cap'n Proto device context data and provides access to camera models,
rigid transforms, and XY lookup tables.
)doc";

constexpr const char* doc_add_device_context = R"doc(
Add a device context for a camera.

Parameters
----------
camera_name : str
    Name of the camera (e.g., "camera0").
device_context : DecodedDeviceContext
    Decoded Cap'n Proto device context.
)doc";

constexpr const char* doc_has_camera = R"doc(
Check if a camera is registered.
)doc";

constexpr const char* doc_camera_names = R"doc(
Get list of registered camera names.
)doc";

constexpr const char* doc_get_camera_name_from_port_name = R"doc(
Extract the camera name from a port name such as ``camera0_depth``.
)doc";

constexpr const char* doc_get_depth_camera_model = R"doc(
Get the depth camera model for a given camera.

Returns ``None`` if the camera is not registered.
)doc";

constexpr const char* doc_get_color_camera_model = R"doc(
Get the color camera model for a given camera.

Returns ``None`` if the camera is not registered.
)doc";

constexpr const char* doc_get_xy_table_intrinsics = R"doc(
Get XY lookup-table intrinsics for a given camera.

Returns a ``pyxylt.IntrinsicParameters`` instance or ``None`` if the camera
is not registered.
)doc";

constexpr const char* doc_get_xy_table = R"doc(
Generate the XY lookup table for a given camera.

Returns a NumPy array of shape ``(height, width, 2)`` or ``None`` if the
camera is not registered.
)doc";

constexpr const char* doc_get_depth_extrinsics = R"doc(
Get the depth camera extrinsics for a given camera.

Returns ``None`` if the camera is not registered.
)doc";

constexpr const char* doc_get_color_to_depth = R"doc(
Get the color-to-depth transform for a given camera.

Returns ``None`` if the camera is not registered.
)doc";

constexpr const char* doc_get_color_to_depth_inv = R"doc(
Get the inverse color-to-depth transform for a given camera.

Returns ``None`` if the camera is not registered.
)doc";

}  // namespace DeviceContextService

namespace XYLookupTableSourceOp {
constexpr const char* doc_XYLookupTableSourceOp = R"doc(
One-shot source operator that emits XY lookup tables for depth reprojection.

Uses DeviceContextService to generate the XY lookup table for a given camera
and emits it as a GPU tensor of shape (height, width, 2).

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment this operator belongs to.
camera_name : str
    Name of the camera to generate XY lookup table for.
allocator : holoscan.resources.Allocator
    GPU memory allocator.
name : str, optional
    Name of the operator. Default: "xy_lookup_table_source".
)doc";

constexpr const char* doc_initialize = "Initialize the operator.";
constexpr const char* doc_setup = "Set up the operator specification.";

constexpr const char* doc_set_device_context_service = R"doc(
Set the device context service.

Parameters
----------
service : DeviceContextService
    The device context service instance.
)doc";

}  // namespace XYLookupTableSourceOp

}  // namespace doc
