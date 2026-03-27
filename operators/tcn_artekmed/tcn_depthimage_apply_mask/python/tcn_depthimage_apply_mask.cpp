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

#include "../tcn_depthimage_apply_mask.cuh"
#include "./tcn_depthimage_apply_mask_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnDepthImageApplyMaskOp : public TcnDepthImageApplyMaskOp {
 public:
  using TcnDepthImageApplyMaskOp::TcnDepthImageApplyMaskOp;

  PyTcnDepthImageApplyMaskOp(const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
                             const py::args& args,
                             std::shared_ptr<holoscan::Allocator> allocator,
                             bool invert_mask = false,
                             const std::string& out_tensor_name = "",
                             const std::string& name = "tcn_depthimage_apply_mask")
      : TcnDepthImageApplyMaskOp(
            holoscan::ArgList{holoscan::Arg{"allocator", allocator},
                              holoscan::Arg{"invert_mask", invert_mask},
                              holoscan::Arg{"out_tensor_name", out_tensor_name}}) {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

PYBIND11_MODULE(_tcn_depthimage_apply_mask, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK TCN DepthImage Apply Mask Python Bindings
        ---------------------------------------
        .. currentmodule:: _tcn_depthimage_apply_mask
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TcnDepthImageApplyMaskOp,
             PyTcnDepthImageApplyMaskOp,
             holoscan::Operator,
             std::shared_ptr<TcnDepthImageApplyMaskOp>>(
      m,
      "TcnDepthImageApplyMaskOp",
      doc::TcnDepthImageApplyMaskOp::doc_TcnDepthImageApplyMaskOp)
      .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                    const py::args&,
                    std::shared_ptr<holoscan::Allocator>,
                    bool,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "invert_mask"_a = false,
           "out_tensor_name"_a = ""s,
           "name"_a = "tcn_depthimage_apply_mask"s,
           doc::TcnDepthImageApplyMaskOp::doc_TcnDepthImageApplyMaskOp)
      .def("setup",
           &TcnDepthImageApplyMaskOp::setup,
           "spec"_a,
           doc::TcnDepthImageApplyMaskOp::doc_setup);

  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    HOLOSCAN_LOG_DEBUG("TCN DepthImage Apply Mask - register types");
  });
}  // PYBIND11_MODULE
}  // namespace tcn::ops
