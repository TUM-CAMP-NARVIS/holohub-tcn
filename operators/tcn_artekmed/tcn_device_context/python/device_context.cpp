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
#include <pybind11/numpy.h>
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

#include "../device_context_service.hpp"
#include "../xy_lookup_table_source_op.hpp"
#include "./device_context_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

namespace {

float dict_float(const py::dict& dict, const char* key) {
    return py::cast<float>(dict[py::str(key)]);
}

bool dict_bool(const py::dict& dict, const char* key) {
    return py::cast<bool>(dict[py::str(key)]);
}

int dict_int(const py::dict& dict, const char* key) {
    return py::cast<int>(dict[py::str(key)]);
}

uint64_t dict_uint64(const py::dict& dict, const char* key) {
    return py::cast<uint64_t>(dict[py::str(key)]);
}


py::dict dict_at(const py::dict& dict, const char* key) {
    return py::cast<py::dict>(dict[py::str(key)]);
}

RigidTransform rigid_transform_from_dict(const py::dict& pose) {
    const auto translation = dict_at(pose, "translation");
    const auto rotation = dict_at(pose, "rotation");

    return RigidTransform{
        Eigen::Vector3f(dict_float(translation, "x"),
                        dict_float(translation, "y"),
                        dict_float(translation, "z")),
        Eigen::Quaternionf(dict_float(rotation, "w"),
                           dict_float(rotation, "x"),
                           dict_float(rotation, "y"),
                           dict_float(rotation, "z"))};
}

nvidia::gxf::CameraModel camera_model_from_dict(const py::dict& params) {
    nvidia::gxf::CameraModel model;
    model.distortion_type = nvidia::gxf::DistortionType::Brown;
    model.dimensions.x = dict_int(params, "width");
    model.dimensions.y = dict_int(params, "height");
    model.focal_length.x = dict_float(params, "fovX");
    model.focal_length.y = dict_float(params, "fovY");
    model.principal_point.x = dict_float(params, "cX");
    model.principal_point.y = dict_float(params, "cY");
    model.skew_value = 1.0f;

    const auto distortion = dict_at(params, "distortionParams");
    model.distortion_coefficients = {
        dict_float(distortion, "k1"),
        dict_float(distortion, "k2"),
        dict_float(distortion, "tx"),
        dict_float(distortion, "ty"),
        dict_float(distortion, "k3"),
        dict_float(distortion, "k4"),
        dict_float(distortion, "k5"),
        dict_float(distortion, "k6"),
    };

    return model;
}

tcn::shm::CameraDeviceInfo camera_device_info_from_dict(const py::dict& context) {
    const auto calibration = dict_at(context, "calibration");

    tcn::shm::CameraDeviceInfo info;
    info.depth_units_per_meter = dict_float(context, "depthUnitsPerMeter");
    info.is_valid = dict_bool(context, "isValid");
    info.frame_rate = dict_float(context, "frameRate");
    info.depth_camera_model =
        camera_model_from_dict(dict_at(calibration, "depthCameraParameters"));
    info.color_camera_model =
        camera_model_from_dict(dict_at(calibration, "colorCameraParameters"));
    info.camera_pose = rigid_transform_from_dict(dict_at(calibration, "cameraPose"));
    info.color_to_depth =
        rigid_transform_from_dict(dict_at(calibration, "color2depthTransform"));
    return info;
}

std::shared_ptr<DeviceContextService> make_device_context_service(
    const py::dict& device_contexts) {
    auto service = std::make_shared<DeviceContextService>();
    for (const auto& item : device_contexts) {
        service->add_device_context(py::cast<std::string>(item.first),
                                    camera_device_info_from_dict(py::cast<py::dict>(item.second)));
    }
    return service;
}

void ensure_camera_bindings_loaded() {
    py::module_::import("holohub.tcn_depthimage_backprojection._tcn_depthimage_backprojection");
}

void ensure_xylt_bindings_loaded() {
    py::module_::import("pyxylt");
}

py::object py_none_or_camera_model(const std::optional<nvidia::gxf::CameraModel>& model) {
    if (!model) {
        return py::none();
    }
    ensure_camera_bindings_loaded();
    return py::cast(*model);
}

py::object py_none_or_rigid_transform(const std::optional<RigidTransform>& transform) {
    if (!transform) {
        return py::none();
    }
    ensure_camera_bindings_loaded();
    return py::cast(*transform);
}

py::object py_none_or_intrinsics(const std::optional<IntrinsicParameters>& intrinsics) {
    if (!intrinsics) {
        return py::none();
    }

    ensure_xylt_bindings_loaded();
    auto intrinsic_parameters = py::module_::import("pyxylt").attr("IntrinsicParameters")();
    intrinsic_parameters.attr("fov_x") = intrinsics->fov_x;
    intrinsic_parameters.attr("fov_y") = intrinsics->fov_y;
    intrinsic_parameters.attr("c_x") = intrinsics->c_x;
    intrinsic_parameters.attr("c_y") = intrinsics->c_y;
    intrinsic_parameters.attr("width") = intrinsics->width;
    intrinsic_parameters.attr("height") = intrinsics->height;
    intrinsic_parameters.attr("tangential_distortion") = py::cast(intrinsics->tangential_distortion);
    intrinsic_parameters.attr("radial_distortion") = py::cast(intrinsics->radial_distortion);
    return intrinsic_parameters;
}

py::object py_none_or_xy_table(const std::shared_ptr<XYTableData>& xy_table) {
    if (!xy_table) {
        return py::none();
    }

    py::array_t<float> array(
        {static_cast<py::ssize_t>(xy_table->height),
         static_cast<py::ssize_t>(xy_table->width),
         py::ssize_t{2}});
    auto view = array.mutable_unchecked<3>();
    size_t index = 0;
    for (size_t row = 0; row < xy_table->height; ++row) {
        for (size_t col = 0; col < xy_table->width; ++col) {
            view(row, col, 0) = xy_table->data[index++];
            view(row, col, 1) = xy_table->data[index++];
        }
    }
    return array;
}

}  // namespace

class PyXYLookupTableSourceOp : public XYLookupTableSourceOp {
 public:
    using XYLookupTableSourceOp::XYLookupTableSourceOp;

    PyXYLookupTableSourceOp(
        const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
        const py::args& args,
        std::shared_ptr<holoscan::Allocator> allocator,
        const std::string& camera_name = "",
        const std::string& name = "xy_lookup_table_source")
        : XYLookupTableSourceOp(
              holoscan::ArgList{
                  holoscan::Arg{"allocator", allocator},
                  holoscan::Arg{"camera_name", camera_name}}) {
        add_positional_condition_and_resource_args(this, args);
        init_operator_base(this, fragment_or_subgraph, name);
    }
};

PYBIND11_MODULE(_tcn_device_context, m) {
    m.doc() = R"pbdoc(
        Holoscan SDK TCN Device Context Python Bindings
        ------------------------------------------------
        .. currentmodule:: _tcn_device_context
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // DeviceContextService binding
    py::class_<DeviceContextService, holoscan::DefaultFragmentService,
               std::shared_ptr<DeviceContextService>>(
        m,
        "DeviceContextService",
        doc::DeviceContextService::doc_DeviceContextService)
        .def(py::init<>())
        .def(py::init([](const py::dict& device_contexts) {
                 return make_device_context_service(device_contexts);
             }),
             "device_contexts"_a,
             "Create a service from the legacy Python device-context dictionary")
        .def_static("create",
                    [](const py::dict& device_contexts) {
                        return make_device_context_service(device_contexts);
                    },
                    "device_contexts"_a,
                    "Create a service from the legacy Python device-context dictionary")
        .def("has_camera",
             &DeviceContextService::has_camera,
             "camera_name"_a,
             doc::DeviceContextService::doc_has_camera)
        .def("camera_names",
             &DeviceContextService::camera_names,
             doc::DeviceContextService::doc_camera_names)
        .def("get_camera_name_from_port_name",
             &DeviceContextService::get_camera_name_from_port_name,
             "port_name"_a,
             doc::DeviceContextService::doc_get_camera_name_from_port_name)
        .def("get_depth_camera_model",
             [](const DeviceContextService& service, const std::string& camera_name) {
                 return py_none_or_camera_model(service.get_depth_camera_model(camera_name));
             },
             "camera_name"_a,
             doc::DeviceContextService::doc_get_depth_camera_model)
        .def("get_color_camera_model",
             [](const DeviceContextService& service, const std::string& camera_name) {
                 return py_none_or_camera_model(service.get_color_camera_model(camera_name));
             },
             "camera_name"_a,
             doc::DeviceContextService::doc_get_color_camera_model)
        .def("get_xy_table_intrinsics",
             [](const DeviceContextService& service, const std::string& camera_name) {
                 return py_none_or_intrinsics(service.get_xy_table_intrinsics(camera_name));
             },
             "camera_name"_a,
             doc::DeviceContextService::doc_get_xy_table_intrinsics)
        .def("get_xy_table",
             [](const DeviceContextService& service, const std::string& camera_name) {
                 return py_none_or_xy_table(service.get_xy_table(camera_name));
             },
             "camera_name"_a,
             doc::DeviceContextService::doc_get_xy_table)
        .def("get_depth_extrinsics",
             [](const DeviceContextService& service, const std::string& camera_name) {
                 return py_none_or_rigid_transform(service.get_depth_extrinsics(camera_name));
             },
             "camera_name"_a,
             doc::DeviceContextService::doc_get_depth_extrinsics)
        .def("get_color_to_depth",
             [](const DeviceContextService& service, const std::string& camera_name) {
                 return py_none_or_rigid_transform(service.get_color_to_depth(camera_name));
             },
             "camera_name"_a,
             doc::DeviceContextService::doc_get_color_to_depth)
        .def("get_color_to_depth_inv",
             [](const DeviceContextService& service, const std::string& camera_name) {
                 return py_none_or_rigid_transform(service.get_color_to_depth_inv(camera_name));
             },
             "camera_name"_a,
             doc::DeviceContextService::doc_get_color_to_depth_inv);


    // XYLookupTableSourceOp binding
    py::class_<XYLookupTableSourceOp,
               PyXYLookupTableSourceOp,
               holoscan::Operator,
               std::shared_ptr<XYLookupTableSourceOp>>(
        m,
        "XYLookupTableSourceOp",
        doc::XYLookupTableSourceOp::doc_XYLookupTableSourceOp)
        .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                      const py::args&,
                      std::shared_ptr<holoscan::Allocator>,
                      const std::string&,
                      const std::string&>(),
             "fragment"_a,
             "allocator"_a,
             "camera_name"_a = ""s,
             "name"_a = "xy_lookup_table_source"s,
             doc::XYLookupTableSourceOp::doc_XYLookupTableSourceOp)
        .def("initialize",
             &XYLookupTableSourceOp::initialize,
             doc::XYLookupTableSourceOp::doc_initialize)
        .def("setup",
             &XYLookupTableSourceOp::setup,
             "spec"_a,
             doc::XYLookupTableSourceOp::doc_setup)
        .def("set_device_context_service",
             &XYLookupTableSourceOp::set_device_context_service,
             "service"_a,
             doc::XYLookupTableSourceOp::doc_set_device_context_service);

    m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
        HOLOSCAN_LOG_DEBUG("TCN Device Context - register types");
    });
}  // PYBIND11_MODULE NOLINT

}  // namespace tcn::ops
