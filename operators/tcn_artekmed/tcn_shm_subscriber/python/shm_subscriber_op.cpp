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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/subgraph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/python/core/component_util.hpp"
#include <holoscan/python/core/emitter_receiver_registry.hpp>

#include "iox2/iceoryx2.hpp"

#include "../shm_subscriber_op.hpp"
#include "./shm_subscriber_op_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

namespace {

std::shared_ptr<tcn::shm::ShmSynchronizedBufferReceiver> make_receiver() {
    auto node = iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().value();
    return std::make_shared<tcn::shm::ShmSynchronizedBufferReceiver>(std::move(node));
}

py::dict rigid_transform_to_dict(const RigidTransform& pose) {
    py::dict rotation;
    rotation["x"] = pose.rotation.x();
    rotation["y"] = pose.rotation.y();
    rotation["z"] = pose.rotation.z();
    rotation["w"] = pose.rotation.w();

    py::dict translation;
    translation["x"] = pose.translation.x();
    translation["y"] = pose.translation.y();
    translation["z"] = pose.translation.z();

    py::dict result;
    result["translation"] = std::move(translation);
    result["rotation"] = std::move(rotation);
    return result;
}

py::dict camera_model_to_dict(const nvidia::gxf::CameraModel& model) {
    py::dict distortion_params;
    distortion_params["k1"] = model.distortion_coefficients[0];
    distortion_params["k2"] = model.distortion_coefficients[1];
    distortion_params["tx"] = model.distortion_coefficients[2];
    distortion_params["ty"] = model.distortion_coefficients[3];
    distortion_params["k3"] = model.distortion_coefficients[4];
    distortion_params["k4"] = model.distortion_coefficients[5];
    distortion_params["k5"] = model.distortion_coefficients[6];
    distortion_params["k6"] = model.distortion_coefficients[7];

    py::dict result;
    result["width"] = model.dimensions.x;
    result["height"] = model.dimensions.y;
    result["fovX"] = model.focal_length.x;
    result["fovY"] = model.focal_length.y;
    result["cX"] = model.principal_point.x;
    result["cY"] = model.principal_point.y;
    result["distortionParams"] = std::move(distortion_params);
    return result;
}

py::dict camera_device_info_to_dict(const tcn::shm::CameraDeviceInfo& info) {
    py::dict calibration;
    calibration["depthCameraParameters"] = camera_model_to_dict(info.depth_camera_model);
    calibration["colorCameraParameters"] = camera_model_to_dict(info.color_camera_model);
    calibration["cameraPose"] = rigid_transform_to_dict(info.camera_pose);
    calibration["color2depthTransform"] = rigid_transform_to_dict(info.color_to_depth);

    py::dict result;
    result["depthUnitsPerMeter"] = info.depth_units_per_meter;
    result["isValid"] = info.is_valid;
    result["frameRate"] = info.frame_rate;
    result["calibration"] = std::move(calibration);
    return result;
}

py::dict channel_port_info_to_dict(const tcn::shm::ChannelPortInfo& info) {
    py::dict buffer_info;
    buffer_info["width"] = info.width;
    buffer_info["height"] = info.height;
    buffer_info["bitsPerElement"] = info.bits_per_element;
    buffer_info["frameSize"] = info.frame_size;
    buffer_info["semanticType"] = info.semantic_type;

    py::dict status;
    status["portType"] = info.port_type;
    status["bufferInfo"] = std::move(buffer_info);

    py::dict result;
    result["name"] = info.name;
    result["status"] = std::move(status);
    return result;
}

py::dict channel_config_to_dict(const std::vector<tcn::shm::ChannelPortInfo>& ports) {
    py::list port_list;
    for (const auto& port : ports) {
        port_list.append(channel_port_info_to_dict(port));
    }

    py::dict result;
    result["ports"] = std::move(port_list);
    return result;
}

py::dict discover_shm(const std::string& stream_name) {
    auto receiver = make_receiver();

    py::dict device_contexts;
    py::list camera_names;
    for (const auto& camera_name : tcn::shm::ShmSynchronizedBufferReceiver::discover_devices()) {
        camera_names.append(camera_name);
        device_contexts[py::str(camera_name)] =
            camera_device_info_to_dict(receiver->retrieve_device_context(camera_name));
    }

    py::dict result;
    result["receiver"] = receiver;
    result["camera_names"] = std::move(camera_names);
    result["device_contexts"] = std::move(device_contexts);
    result["channels_config"] = channel_config_to_dict(receiver->retrieve_channel_config(stream_name));
    return result;
}

}  // namespace

class PyTcnShmSubscriberOp : public TcnShmSubscriberOp {
 public:
    using TcnShmSubscriberOp::TcnShmSubscriberOp;

    PyTcnShmSubscriberOp(
        const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
        const py::args& args,
        std::shared_ptr<holoscan::Allocator> allocator,
        std::shared_ptr<holoscan::AsynchronousCondition> async_condition,
        const std::string& stream_name = "",
        int32_t cycle_time_ms = 1,
        const std::string& name = "tcn_shm_subscriber")
        : TcnShmSubscriberOp(
              holoscan::ArgList{
                  holoscan::Arg{"allocator", allocator},
                  holoscan::Arg{"async_condition", async_condition},
                  holoscan::Arg{"stream_name", stream_name},
                  holoscan::Arg{"cycle_time_ms", cycle_time_ms}}) {
        add_positional_condition_and_resource_args(this, args);
        init_operator_base(this, fragment_or_subgraph, name);
    }

    PyTcnShmSubscriberOp(
        const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
        const py::args& args,
        std::shared_ptr<holoscan::Allocator> allocator,
        std::shared_ptr<holoscan::AsynchronousCondition> async_condition,
        std::shared_ptr<tcn::shm::ShmSynchronizedBufferReceiver> receiver,
        const std::string& stream_name = "",
        int32_t cycle_time_ms = 1,
        const std::string& name = "tcn_shm_subscriber")
        : PyTcnShmSubscriberOp(fragment_or_subgraph,
                               args,
                               std::move(allocator),
                               std::move(async_condition),
                               stream_name,
                               cycle_time_ms,
                               name) {
        set_receiver(std::move(receiver));
    }
};

PYBIND11_MODULE(_tcn_shm_subscriber, m) {
    m.doc() = R"pbdoc(
        Holoscan SDK TCN SHM Subscriber Python Bindings
        -----------------------------------------------
        .. currentmodule:: _tcn_shm_subscriber
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<TcnShmSubscriberOp,
               PyTcnShmSubscriberOp,
               holoscan::Operator,
               std::shared_ptr<TcnShmSubscriberOp>>(
        m,
        "TcnShmSubscriberOp",
        doc::TcnShmSubscriberOp::doc_TcnShmSubscriberOp)
        .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                      const py::args&,
                      std::shared_ptr<holoscan::Allocator>,
                      std::shared_ptr<holoscan::AsynchronousCondition>,
                      const std::string&,
                      int32_t,
                      const std::string&>(),
             "fragment"_a,
             "allocator"_a,
             "async_condition"_a,
             "stream_name"_a = ""s,
             "cycle_time_ms"_a = 1,
             "name"_a = "tcn_shm_subscriber"s,
             doc::TcnShmSubscriberOp::doc_TcnShmSubscriberOp)
        .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                      const py::args&,
                      std::shared_ptr<holoscan::Allocator>,
                      std::shared_ptr<holoscan::AsynchronousCondition>,
                      std::shared_ptr<tcn::shm::ShmSynchronizedBufferReceiver>,
                      const std::string&,
                      int32_t,
                      const std::string&>(),
             "fragment"_a,
             "allocator"_a,
             "async_condition"_a,
             "receiver"_a,
             "stream_name"_a = ""s,
             "cycle_time_ms"_a = 1,
             "name"_a = "tcn_shm_subscriber"s,
             doc::TcnShmSubscriberOp::doc_TcnShmSubscriberOp)
        .def("initialize",
             &TcnShmSubscriberOp::initialize,
             doc::TcnShmSubscriberOp::doc_initialize)
        .def("setup",
             &TcnShmSubscriberOp::setup,
             "spec"_a,
             doc::TcnShmSubscriberOp::doc_setup)
        .def("set_receiver",
             &TcnShmSubscriberOp::set_receiver,
             "receiver"_a,
             "Set the SHM receiver instance");

    // Expose ShmSynchronizedBufferReceiver for Python use
    py::class_<tcn::shm::ShmSynchronizedBufferReceiver,
               std::shared_ptr<tcn::shm::ShmSynchronizedBufferReceiver>>(
        m, "ShmSynchronizedBufferReceiver",
        doc::ShmSynchronizedBufferReceiver::doc_ShmSynchronizedBufferReceiver)
        .def_static("create",
                     []() { return make_receiver(); },
                     "Create a receiver with an internally owned iceoryx2 IPC node")
        .def_static("discover_devices",
                     &tcn::shm::ShmSynchronizedBufferReceiver::discover_devices,
                     doc::ShmSynchronizedBufferReceiver::doc_discover_devices)
        .def("retrieve_device_context",
             [](tcn::shm::ShmSynchronizedBufferReceiver& self, const std::string& camera_name) {
                 return camera_device_info_to_dict(self.retrieve_device_context(camera_name));
             },
             "camera_name"_a,
             "Retrieve device context as a Python dict compatible with the legacy API")
        .def("retrieve_channel_config",
             [](tcn::shm::ShmSynchronizedBufferReceiver& self, const std::string& stream_name) {
                 return channel_config_to_dict(self.retrieve_channel_config(stream_name));
             },
             "stream_name"_a,
             "Retrieve channel configuration as a Python dict compatible with the legacy API")
        .def("subscribe",
             &tcn::shm::ShmSynchronizedBufferReceiver::subscribe,
             "stream_name"_a,
             "Subscribe to the shared-memory frame stream")
        .def("teardown",
             &tcn::shm::ShmSynchronizedBufferReceiver::teardown,
             "Tear down the shared-memory receiver");

    m.def("discover_shm",
          &discover_shm,
          "stream_name"_a,
          "Discover cameras and channel configuration for a shared-memory stream");
    m.def("create_receiver",
          []() { return make_receiver(); },
          "Create a shared-memory receiver with an internally owned iceoryx2 IPC node");

    m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
        HOLOSCAN_LOG_DEBUG("TCN SHM Subscriber - register types");
    });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
