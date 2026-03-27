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

#include "../tcn_convert_bgra_to_rgba.cuh"
#include "./tcn_convert_bgra_to_rgba_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnConvertBgraToRgbaOp : public TcnConvertBgraToRgbaOp {
 public:
  using TcnConvertBgraToRgbaOp::TcnConvertBgraToRgbaOp;

  PyTcnConvertBgraToRgbaOp(const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
                           const py::args& args,
                           std::shared_ptr<holoscan::Allocator> allocator,
                           const std::string& in_tensor_name = "",
                           const std::string& out_tensor_name = "",
                           const std::string& name = "tcn_convert_bgra_to_rgba")
      : TcnConvertBgraToRgbaOp(
            holoscan::ArgList{holoscan::Arg{"allocator", allocator},
                              holoscan::Arg{"in_tensor_name", in_tensor_name},
                              holoscan::Arg{"out_tensor_name", out_tensor_name}}) {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

PYBIND11_MODULE(_tcn_convert_bgra_to_rgba, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK TCN Convert BGRA to RGBA Python Bindings
        ---------------------------------------
        .. currentmodule:: _tcn_convert_bgra_to_rgba
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TcnConvertBgraToRgbaOp,
             PyTcnConvertBgraToRgbaOp,
             holoscan::Operator,
             std::shared_ptr<TcnConvertBgraToRgbaOp>>(
      m,
      "TcnConvertBgraToRgbaOp",
      doc::TcnConvertBgraToRgbaOp::doc_TcnConvertBgraToRgbaOp)
      .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                    const py::args&,
                    std::shared_ptr<holoscan::Allocator>,
                    const std::string&,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "in_tensor_name"_a = ""s,
           "out_tensor_name"_a = ""s,
           "name"_a = "tcn_convert_bgra_to_rgba"s,
           doc::TcnConvertBgraToRgbaOp::doc_TcnConvertBgraToRgbaOp)
      .def("setup",
           &TcnConvertBgraToRgbaOp::setup,
           "spec"_a,
           doc::TcnConvertBgraToRgbaOp::doc_setup);

  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    HOLOSCAN_LOG_DEBUG("TCN Convert BGRA to RGBA - register types");
  });
}  // PYBIND11_MODULE
}  // namespace tcn::ops
