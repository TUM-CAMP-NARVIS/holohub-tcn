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

#include <cstdint>
#include <memory>
#include <string>
#include <variant>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/subgraph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/python/core/component_util.hpp"
#include <holoscan/python/core/emitter_receiver_registry.hpp>

#include "../tcn_depthimage_fgbg_mask.cuh"
#include "./tcn_depthimage_fgbg_mask_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnDepthImageFgbgMaskOp : public TcnDepthImageFgbgMaskOp {
 public:
  using TcnDepthImageFgbgMaskOp::TcnDepthImageFgbgMaskOp;

  PyTcnDepthImageFgbgMaskOp(const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
                            const py::args& args,
                            std::shared_ptr<holoscan::Allocator> allocator,
                            float sensitivity = 1.0f,
                            bool enable_foreground = true,
                            bool enable_background = false,
                            const std::string& name = "tcn_depthimage_fgbg_mask")
      : TcnDepthImageFgbgMaskOp(
            holoscan::ArgList{holoscan::Arg{"allocator", allocator},
                              holoscan::Arg{"sensitivity", sensitivity},
                              holoscan::Arg{"enable_foreground", enable_foreground},
                              holoscan::Arg{"enable_background", enable_background}}) {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

PYBIND11_MODULE(_tcn_depthimage_fgbg_mask, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK TCN DepthImage FG/BG Mask Python Bindings
        ---------------------------------------
        .. currentmodule:: _tcn_depthimage_fgbg_mask
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TcnDepthImageFgbgMaskOp,
             PyTcnDepthImageFgbgMaskOp,
             holoscan::Operator,
             std::shared_ptr<TcnDepthImageFgbgMaskOp>>(
      m,
      "TcnDepthImageFgbgMaskOp",
      doc::TcnDepthImageFgbgMaskOp::doc_TcnDepthImageFgbgMaskOp)
      .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                    const py::args&,
                    std::shared_ptr<holoscan::Allocator>,
                    float,
                    bool,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "sensitivity"_a = 1.0f,
           "enable_foreground"_a = true,
           "enable_background"_a = false,
           "name"_a = "tcn_depthimage_fgbg_mask"s,
           doc::TcnDepthImageFgbgMaskOp::doc_TcnDepthImageFgbgMaskOp)
      .def("setup",
           &TcnDepthImageFgbgMaskOp::setup,
           "spec"_a,
           doc::TcnDepthImageFgbgMaskOp::doc_setup);

  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    HOLOSCAN_LOG_DEBUG("TCN DepthImage FG/BG Mask - register types");
  });
}  // PYBIND11_MODULE
}  // namespace tcn::ops
