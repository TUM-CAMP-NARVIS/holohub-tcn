// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cdr_type_registry.hpp"

#include <tcnart_msgs/msg/VideoStream.h>
#include <tcnart_msgs/msg/StreamDescriptor.h>
#include <tcnart_msgs/msg/PoseTracking.h>
#include <tcnart_msgs/msg/CameraModel.h>
#include <tcnart_msgs/msg/AudioStream.h>
#include <tcnart_msgs/msg/Logging.h>
#include <tcnart_msgs/msg/Presence.h>
#include <tcnart_msgs/msg/HandTracking.h>
#include <tcnart_msgs/msg/EyeTracking.h>
#include <tcnart_msgs/msg/SpatialRelations.h>
#include <tcnart_msgs/msg/XRInput.h>
#include <tcnart_msgs/msg/XRRuntime.h>

namespace tcn::cdr {
namespace {

/// Helper: deserialize a CDR message into the given type using CdrBufferReader.
template <typename MT>
bool deserialize_cdr(const uint8_t* data, size_t size, MT& msg) {
    CdrBufferReader reader;
    return reader.read(data, size, msg);
}

// --- VideoStreamMessage ---
static CdrTypeRegistrar reg_video_stream(
    "tcnart_msgs::msg::VideoStreamMessage",
    [](const uint8_t* data, size_t size, DecodedMessage& out) -> bool {
        tcnart_msgs::msg::VideoStreamMessage msg;
        if (!deserialize_cdr(data, size, msg)) return false;
        auto& img = msg.image();
        out.payload.assign(img.begin(), img.end());
        out.metadata["image_bytes"] = std::to_string(msg.image_bytes());
        auto& header = msg.header();
        out.metadata["frame_id"] = header.frame_id();
        out.metadata["stamp_sec"] = std::to_string(header.stamp().sec());
        out.metadata["stamp_nanosec"] = std::to_string(header.stamp().nanosec());
        return true;
    });

// --- StreamDescriptorMessage ---
static CdrTypeRegistrar reg_stream_descriptor(
    "tcnart_msgs::msg::StreamDescriptorMessage",
    [](const uint8_t* data, size_t size, DecodedMessage& out) -> bool {
        tcnart_msgs::msg::StreamDescriptorMessage msg;
        if (!deserialize_cdr(data, size, msg)) return false;
        // No bulk payload for descriptors; metadata carries the fields.
        out.metadata["stream_topic"] = msg.stream_topic();
        out.metadata["stream_topic_schema"] = msg.stream_topic_schema();
        out.metadata["calib_topic"] = msg.calib_topic();
        out.metadata["calib_topic_schema"] = msg.calib_topic_schema();
        out.metadata["sensor_type"] = std::to_string(static_cast<int>(msg.sensor_type()));
        out.metadata["frame_rate"] = std::to_string(msg.frame_rate());
        out.metadata["image_height"] = std::to_string(msg.image_height());
        out.metadata["image_width"] = std::to_string(msg.image_width());
        out.metadata["image_step"] = std::to_string(msg.image_step());
        out.metadata["image_format"] = std::to_string(static_cast<int>(msg.image_format()));
        out.metadata["image_compression"] = std::to_string(static_cast<int>(msg.image_compression()));
        auto& header = msg.header();
        out.metadata["frame_id"] = header.frame_id();
        return true;
    });

// --- Pose6DMessage ---
static CdrTypeRegistrar reg_pose6d(
    "tcnart_msgs::msg::Pose6DMessage",
    [](const uint8_t* data, size_t size, DecodedMessage& out) -> bool {
        tcnart_msgs::msg::Pose6DMessage msg;
        if (!deserialize_cdr(data, size, msg)) return false;
        auto& t = msg.value().translation();
        auto& r = msg.value().rotation();
        out.metadata["tx"] = std::to_string(t.x());
        out.metadata["ty"] = std::to_string(t.y());
        out.metadata["tz"] = std::to_string(t.z());
        out.metadata["rx"] = std::to_string(r.x());
        out.metadata["ry"] = std::to_string(r.y());
        out.metadata["rz"] = std::to_string(r.z());
        out.metadata["rw"] = std::to_string(r.w());
        auto& header = msg.header();
        out.metadata["frame_id"] = header.frame_id();
        return true;
    });

// --- Pose6DMapMessage ---
static CdrTypeRegistrar reg_pose6d_map(
    "tcnart_msgs::msg::Pose6DMapMessage",
    [](const uint8_t* data, size_t size, DecodedMessage& out) -> bool {
        tcnart_msgs::msg::Pose6DMapMessage msg;
        if (!deserialize_cdr(data, size, msg)) return false;
        auto& header = msg.header();
        out.metadata["frame_id"] = header.frame_id();
        // Serialize the transform map keys/values into metadata.
        auto& tmap = msg.value();
        out.metadata["pose_count"] = std::to_string(tmap.poses().size());
        return true;
    });

// --- CameraInfoMessage ---
static CdrTypeRegistrar reg_camera_info(
    "tcnart_msgs::msg::CameraInfoMessage",
    [](const uint8_t* data, size_t size, DecodedMessage& out) -> bool {
        tcnart_msgs::msg::CameraInfoMessage msg;
        if (!deserialize_cdr(data, size, msg)) return false;
        auto& cam = msg.value();
        out.metadata["camera_model"] = std::to_string(static_cast<int>(cam.camera_model()));
        out.metadata["image_width"] = std::to_string(cam.image_width());
        out.metadata["image_height"] = std::to_string(cam.image_height());
        auto& fl = cam.focal_length();
        out.metadata["focal_length_x"] = std::to_string(fl[0]);
        out.metadata["focal_length_y"] = std::to_string(fl[1]);
        auto& pp = cam.principal_point();
        out.metadata["principal_point_x"] = std::to_string(pp[0]);
        out.metadata["principal_point_y"] = std::to_string(pp[1]);
        auto& header = msg.header();
        out.metadata["frame_id"] = header.frame_id();
        return true;
    });

}  // namespace
}  // namespace tcn::cdr
