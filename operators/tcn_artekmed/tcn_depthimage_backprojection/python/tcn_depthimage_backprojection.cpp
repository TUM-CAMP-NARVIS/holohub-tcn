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

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <cstdint>
#include <memory>
#include <string>
#include <variant>

#include "gxf/multimedia/camera.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/subgraph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/python/core/component_util.hpp"

#include <holoscan/python/core/emitter_receiver_registry.hpp>

#include "../../common/datatypes.hpp"
#include "../tcn_depthimage_backprojection.cuh"
#include "./tcn_depthimage_backprojection_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyTcnDepthImageBackprojectionOp : public TcnDepthImageBackprojectionOp {
 public:
  /* Inherit the constructors */
  using TcnDepthImageBackprojectionOp::TcnDepthImageBackprojectionOp;

  // Define a constructor that fully initializes the object.
  PyTcnDepthImageBackprojectionOp(const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
                                  const py::args& args,
                                  std::shared_ptr<holoscan::Allocator> allocator,
                                  float depth_units_per_meter=1000.f,
                                  float near_limit_m=0.01f,
                                  float far_limit_m=10.f,
                                  int color_image_width=320,
                                  int color_image_height=288,
                                  nvidia::gxf::CameraModel color_params=nvidia::gxf::CameraModel{},
                                  RigidTransform depth_extrinsics=RigidTransform{},
                                  RigidTransform color_to_depth=RigidTransform{},
                                  const std::string& in_tensor_name="",
                                  const std::string& out_tensor_name="",
                                  bool enable_positions = true,
                                  bool enable_texcoords = false,
                                  bool enable_depth_float = false,
                                  int cuda_device_ordinal = 0,
                                  const std::string& name = "tcn_depthimage_backprojection")
      : TcnDepthImageBackprojectionOp(
            holoscan::ArgList{holoscan::Arg{"allocator", allocator},
                              holoscan::Arg{"depth_units_per_meter", depth_units_per_meter},
                              holoscan::Arg{"near_limit_m", near_limit_m},
                              holoscan::Arg{"far_limit_m", far_limit_m},
                              holoscan::Arg{"color_image_width", color_image_width},
                              holoscan::Arg{"color_image_height", color_image_height},
                              holoscan::Arg{"color_params", color_params},
                              holoscan::Arg{"depth_extrinsics", depth_extrinsics},
                              holoscan::Arg{"color_to_depth", color_to_depth},
                              holoscan::Arg{"in_tensor_name", in_tensor_name},
                              holoscan::Arg{"out_tensor_name", out_tensor_name},
                              holoscan::Arg{"enable_positions", enable_positions},
                              holoscan::Arg{"enable_texcoords", enable_texcoords},
                              holoscan::Arg{"enable_depth_float", enable_depth_float},
                              holoscan::Arg{"cuda_device_ordinal", cuda_device_ordinal}
            }) {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

/* The python module */

PYBIND11_MODULE(_tcn_depthimage_backprojection, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK TCN DepthImage Backprojection Python Bindings
        ---------------------------------------
        .. currentmodule:: _tcn_depthimage_backprojection
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::enum_<nvidia::gxf::DistortionType>(
      m, "DistortionType", doc::TcnDepthImageBackprojectionOp::doc_DistortionType)
      .value("Perspective", nvidia::gxf::DistortionType::Perspective)
      .value("Brown", nvidia::gxf::DistortionType::Brown)
      .value("Polynomial", nvidia::gxf::DistortionType::Polynomial)
      .value("FisheyeEquidistant", nvidia::gxf::DistortionType::FisheyeEquidistant)
      .value("FisheyeEquisolid", nvidia::gxf::DistortionType::FisheyeEquisolid)
      .value("FisheyeOrthoGraphic", nvidia::gxf::DistortionType::FisheyeOrthoGraphic)
      .value("FisheyeStereographic", nvidia::gxf::DistortionType::FisheyeStereographic);

  py::class_<nvidia::gxf::Vector2u>(m, "Vector2u", doc::TcnDepthImageBackprojectionOp::doc_Vector2u)
      .def(py::init<>())
      .def_readwrite("x", &nvidia::gxf::Vector2u::x)
      .def_readwrite("y", &nvidia::gxf::Vector2u::y);

  py::class_<nvidia::gxf::Vector2f>(m, "Vector2f", doc::TcnDepthImageBackprojectionOp::doc_Vector2f)
      .def(py::init<>())
      .def_readwrite("x", &nvidia::gxf::Vector2f::x)
      .def_readwrite("y", &nvidia::gxf::Vector2f::y);

  py::class_<CameraParameters>(m, "CameraParameters")
        .def(py::init<>())
        .def_readwrite("fx", &CameraParameters::fx)
        .def_readwrite("fy", &CameraParameters::fy)
        .def_readwrite("cx", &CameraParameters::cx)
        .def_readwrite("cy", &CameraParameters::cy)
        .def_readwrite("k1", &CameraParameters::k1)
        .def_readwrite("k2", &CameraParameters::k2)
        .def_readwrite("k3", &CameraParameters::k3)
        .def_readwrite("k4", &CameraParameters::k4)
        .def_readwrite("k5", &CameraParameters::k5)
        .def_readwrite("k6", &CameraParameters::k6)
        .def_readwrite("codx", &CameraParameters::codx)
        .def_readwrite("cody", &CameraParameters::cody)
        .def_readwrite("p1", &CameraParameters::p1)
        .def_readwrite("p2", &CameraParameters::p2)
        .def_readwrite("is_distorted", &CameraParameters::is_distorted);

  py::class_<nvidia::gxf::CameraModel>(
      m, "CameraModel", doc::TcnDepthImageBackprojectionOp::doc_CameraModel)
      .def(py::init<>())
      .def_readwrite("dimensions", &nvidia::gxf::CameraModel::dimensions)
      .def_readwrite("focal_length", &nvidia::gxf::CameraModel::focal_length)
      .def_readwrite("principal_point", &nvidia::gxf::CameraModel::principal_point)
      .def_readwrite("skew_value", &nvidia::gxf::CameraModel::skew_value)
      .def_readwrite("distortion_type", &nvidia::gxf::CameraModel::distortion_type)
      .def_readwrite("distortion_coefficients", &nvidia::gxf::CameraModel::distortion_coefficients);

  py::class_<RigidTransform>(m, "RigidTransform", doc::TcnDepthImageBackprojectionOp::doc_Pose)
    .def(py::init<Eigen::Vector3f, Eigen::Quaternion<float>>())
    .def_property("translation",
            [](const RigidTransform& p) { return p.translation; },  // getter
            [](RigidTransform& p, const Eigen::Vector3f& v) { p.translation = v; })  // setter
    .def_property("rotation",
        [](const RigidTransform& p) { return p.rotation.coeffs(); },
        // order: x,y,z,w
        [](RigidTransform& p, const Eigen::Vector4f& q) { p.rotation = Eigen::Quaternionf(q.w(), q.x(), q.y(), q.z()); })
    ;

  m.def("make_rigid_transform", [](const Eigen::Vector3f& translation, const Eigen::Vector4f& rotation) {
    return RigidTransform(translation, Eigen::Quaternion<float>(rotation));
  });

  py::class_<TcnDepthImageBackprojectionOp,
             PyTcnDepthImageBackprojectionOp,
             holoscan::Operator,
             std::shared_ptr<TcnDepthImageBackprojectionOp>>(
      m,
      "TcnDepthImageBackprojectionOp",
      doc::TcnDepthImageBackprojectionOp::doc_TcnDepthImageBackprojectionOp)
      .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                    const py::args&,
                    std::shared_ptr<holoscan::Allocator>,
                    float,
                    float,
                    float,
                    int,
                    int,
                    nvidia::gxf::CameraModel,
                    RigidTransform,
                    RigidTransform,
                    const std::string&,
                    const std::string&,
                    bool,
                    bool,
                    bool,
                    int,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "depth_units_per_meter"_a = 1000.f,
           "near_limit_m"_a = 0.01f,
           "far_limit_m"_a = 10.f,
           "color_image_width"_a = 320,
           "color_image_height"_a = 288,
           "color_params"_a = nvidia::gxf::CameraModel{},
           "depth_extrinsics"_a = RigidTransform{},
           "color_to_depth"_a = RigidTransform{},
           "in_tensor_name"_a = ""s,
           "out_tensor_name"_a = ""s,
           "enable_positions"_a = true,
           "enable_texcoords"_a = true,
           "enable_depth_float"_a = false,
           "cuda_device_ordinal"_a = 0,
           "name"_a = "tcn_depthimage_backprojection"s,
           doc::TcnDepthImageBackprojectionOp::doc_TcnDepthImageBackprojectionOp)
      .def("initialize",
           &TcnDepthImageBackprojectionOp::initialize,
           doc::TcnDepthImageBackprojectionOp::doc_initialize)
      .def("setup",
           &TcnDepthImageBackprojectionOp::setup,
           "spec"_a,
           doc::TcnDepthImageBackprojectionOp::doc_setup);

  // Import the emitter/receiver registry from holoscan.core and pass it to this function to
  // register this new C++ type with the SDK.
  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    HOLOSCAN_LOG_DEBUG("TCN SHM Receiver - register types");
    // registry.add_emitter_receiver<nvidia::gxf::CameraModel>(
    //     "nvidia::gxf::CameraModel"s);
    // registry.add_emitter_receiver<holoscan::Pose3f>(
    //     "holoscan::Pose3f"s);
    // should have some reasonable namespacing here ..
    registry.add_emitter_receiver<RigidTransform>(
        "RigidTransform"s);
  });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
