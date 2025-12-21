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

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "gxf/multimedia/camera.hpp"
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
  PyTcnDepthImageBackprojectionOp(holoscan::Fragment* fragment, const py::args& args,
                     std::shared_ptr<::holoscan::Allocator> allocator,
                     float depth_units_per_meter, float near_limit_m, float far_limit_m,
                     int color_image_width, int color_image_height,
                     const std::string& name = "tcn_depthimage_backprojection")
      : TcnDepthImageBackprojectionOp(holoscan::ArgList{
                                 holoscan::Arg{"allocator", allocator},
                                 holoscan::Arg{"depth_units_per_meter", depth_units_per_meter},
                                 holoscan::Arg{"near_limit_m", near_limit_m},
                                 holoscan::Arg{"far_limit_m", far_limit_m},
                                 holoscan::Arg{"color_image_width", color_image_width},
                                 holoscan::Arg{"color_image_height", color_image_height}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_tcn_depthimage_backprojection, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
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

  py::enum_<nvidia::gxf::DistortionType>(m, "DistortionType")
    .value("Perspective", nvidia::gxf::DistortionType::Perspective)
    .value("Brown", nvidia::gxf::DistortionType::Brown)
    .value("Polynomial", nvidia::gxf::DistortionType::Polynomial)
    .value("FisheyeEquidistant", nvidia::gxf::DistortionType::FisheyeEquidistant)
    .value("FisheyeEquisolid", nvidia::gxf::DistortionType::FisheyeEquisolid)
    .value("FisheyeOrthoGraphic", nvidia::gxf::DistortionType::FisheyeOrthoGraphic)
    .value("FisheyeStereographic", nvidia::gxf::DistortionType::FisheyeStereographic);

  py::class_<nvidia::gxf::Vector2u>(m, "Vector2u")
    .def(py::init<>())
    .def_readwrite("x", &nvidia::gxf::Vector2u::x)
    .def_readwrite("y", &nvidia::gxf::Vector2u::y)
  ;
  py::class_<nvidia::gxf::Vector2f>(m, "Vector2f")
    .def(py::init<>())
    .def_readwrite("x", &nvidia::gxf::Vector2f::x)
    .def_readwrite("y", &nvidia::gxf::Vector2f::y)
  ;

  py::class_<nvidia::gxf::CameraModel>(m, "CameraModel")
    .def(py::init<>())
    .def_readwrite("dimensions", &nvidia::gxf::CameraModel::dimensions)
    .def_readwrite("focal_length", &nvidia::gxf::CameraModel::focal_length)
    .def_readwrite("principal_point", &nvidia::gxf::CameraModel::principal_point)
    .def_readwrite("skew_value", &nvidia::gxf::CameraModel::skew_value)
    .def_readwrite("distortion_type", &nvidia::gxf::CameraModel::distortion_type)
    .def_readwrite("distortion_coefficients", &nvidia::gxf::CameraModel::distortion_coefficients)
  ;
  // py::class_<nvidia::gxf::Pose3D>(m, "Pose3D")
  //   .def(py::init<>())
  //   .def_readwrite("rotation", &nvidia::gxf::Pose3D::rotation)
  //   .def_readwrite("translation", &nvidia::gxf::Pose3D::translation)
  // ;

  py::class_<TcnDepthImageBackprojectionOp, PyTcnDepthImageBackprojectionOp, holoscan::Operator, std::shared_ptr<TcnDepthImageBackprojectionOp>>(
      m, "TcnDepthImageBackprojectionOp", doc::TcnDepthImageBackprojectionOp::doc_TcnDepthImageBackprojectionOp)
      .def(py::init<holoscan::Fragment*,
                    const py::args&,
                    std::shared_ptr<::holoscan::Allocator>,
                    float,
                    float,
                    float,
                    int,
                    int,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "depth_units_per_meter"_a = 1000.f,
           "near_limit_m"_a = 0.01f,
           "far_limit_m"_a = 10.f,
           "color_image_width"_a = 1920,
           "color_image_height"_a = 1080,
           "name"_a = "tcn_depthimage_backprojection"s,
           doc::TcnDepthImageBackprojectionOp::doc_TcnDepthImageBackprojectionOp)
      .def("initialize", &TcnDepthImageBackprojectionOp::initialize, doc::TcnDepthImageBackprojectionOp::doc_initialize)
      .def("setup", &TcnDepthImageBackprojectionOp::setup, "spec"_a, doc::TcnDepthImageBackprojectionOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
