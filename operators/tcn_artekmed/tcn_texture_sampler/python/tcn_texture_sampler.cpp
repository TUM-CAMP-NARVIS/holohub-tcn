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

#include "gxf/multimedia/camera.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

#include <holoscan/python/core/emitter_receiver_registry.hpp>

#include "../../common/datatypes.hpp"
#include "../tcn_texture_sampler.cuh"
#include "./tcn_texture_sampler_pydoc.hpp"

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

class PyTcnTextureSamplerOp : public TcnTextureSamplerOp {
 public:
  /* Inherit the constructors */
  using TcnTextureSamplerOp::TcnTextureSamplerOp;

  // Define a constructor that fully initializes the object.
  PyTcnTextureSamplerOp(holoscan::Fragment* fragment, const py::args& args,
                                  std::shared_ptr<holoscan::Allocator> allocator,
                                  int cuda_device_ordinal=0,
                                  const std::string& in_color_tensor_name="",
                                  const std::string& in_texcoord_tensor_name="",
                                  const std::string& out_tensor_name="",
                                  const std::string& name = "tcn_texture_sampler")
      : TcnTextureSamplerOp(
            holoscan::ArgList{holoscan::Arg{"allocator", allocator},
                              holoscan::Arg{"cuda_device_ordinal", cuda_device_ordinal},
                              holoscan::Arg{"in_color_tensor_name", in_color_tensor_name},
                              holoscan::Arg{"in_texcoord_tensor_name", in_texcoord_tensor_name},
                              holoscan::Arg{"out_tensor_name", out_tensor_name}

            }) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_tcn_texture_sampler, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK TCN Texture Sampler Python Bindings
        ---------------------------------------
        .. currentmodule:: _tcn_texture_sampler
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TcnTextureSamplerOp,
             PyTcnTextureSamplerOp,
             holoscan::Operator,
             std::shared_ptr<TcnTextureSamplerOp>>(
      m,
      "TcnTextureSamplerOp",
      doc::TcnTextureSamplerOp::doc_TcnTextureSamplerOp)
      .def(py::init<holoscan::Fragment*,
                    const py::args&,
                    std::shared_ptr<holoscan::Allocator>,
                    int,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "cuda_device_ordinal"_a = 0,
           "in_color_tensor_name"_a = ""s,
           "in_texcoord_tensor_name"_a = ""s,
           "out_tensor_name"_a = ""s,
           "name"_a = "tcn_texture_sampler"s,
           doc::TcnTextureSamplerOp::doc_TcnTextureSamplerOp)
      .def("initialize",
           &TcnTextureSamplerOp::initialize,
           doc::TcnTextureSamplerOp::doc_initialize)
      .def("setup",
           &TcnTextureSamplerOp::setup,
           "spec"_a,
           doc::TcnTextureSamplerOp::doc_setup);

  // Import the emitter/receiver registry from holoscan.core and pass it to this function to
  // register this new C++ type with the SDK.
  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    HOLOSCAN_LOG_DEBUG("TCN Texture Sampler - register types");
  });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
