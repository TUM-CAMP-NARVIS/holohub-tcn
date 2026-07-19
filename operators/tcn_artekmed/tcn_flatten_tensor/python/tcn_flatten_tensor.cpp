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

#include "../tcn_flatten_tensor.cuh"
#include "./tcn_flatten_tensor_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnFlattenTensorOp : public TcnFlattenTensorOp {
 public:
  using TcnFlattenTensorOp::TcnFlattenTensorOp;

  PyTcnFlattenTensorOp(const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
                       const py::args& args,
                       const std::string& message_name = "",
                       const std::string& name = "tcn_flatten_tensor")
      : TcnFlattenTensorOp(
            holoscan::ArgList{holoscan::Arg{"message_name", message_name}}) {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

PYBIND11_MODULE(_tcn_flatten_tensor, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK TCN Flatten Tensor Python Bindings
        ---------------------------------------
        .. currentmodule:: _tcn_flatten_tensor
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TcnFlattenTensorOp,
             PyTcnFlattenTensorOp,
             holoscan::Operator,
             std::shared_ptr<TcnFlattenTensorOp>>(
      m,
      "TcnFlattenTensorOp",
      doc::TcnFlattenTensorOp::doc_TcnFlattenTensorOp)
      .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                    const py::args&,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "message_name"_a = ""s,
           "name"_a = "tcn_flatten_tensor"s,
           doc::TcnFlattenTensorOp::doc_TcnFlattenTensorOp)
      .def("setup",
           &TcnFlattenTensorOp::setup,
           "spec"_a,
           doc::TcnFlattenTensorOp::doc_setup);

  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    HOLOSCAN_LOG_DEBUG("TCN Flatten Tensor - register types");
  });
}  // PYBIND11_MODULE
}  // namespace tcn::ops
