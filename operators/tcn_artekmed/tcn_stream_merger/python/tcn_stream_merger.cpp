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
#include <string>
#include <variant>
#include <vector>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/subgraph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/python/core/component_util.hpp"

#include <holoscan/python/core/emitter_receiver_registry.hpp>

#include "../tcn_stream_merger.hpp"
#include "./tcn_stream_merger_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnStreamMergerOp : public TcnStreamMergerOp {
 public:
  using TcnStreamMergerOp::TcnStreamMergerOp;

  PyTcnStreamMergerOp(
      const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
      const py::args& args,
      const std::vector<std::string>& input_port_names,
      const std::string& input_message_name,
      const std::string& output_message_name,
      bool fuse_buffers = false,
      std::shared_ptr<holoscan::Allocator> allocator = nullptr,
      const std::string& name = "tcn_stream_merger")
      : TcnStreamMergerOp(
            holoscan::ArgList{
                holoscan::Arg{"input_port_names", input_port_names},
                holoscan::Arg{"input_message_name", input_message_name},
                holoscan::Arg{"output_message_name", output_message_name},
                holoscan::Arg{"fuse_buffers", fuse_buffers}}) {
    // Store port names before init_operator_base triggers setup()
    input_port_names_init_ = input_port_names;
    if (allocator) {
      this->add_arg(holoscan::Arg{"allocator", allocator});
    }
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

PYBIND11_MODULE(_tcn_stream_merger, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK TCN Stream Merger Python Bindings
        ---------------------------------------
        .. currentmodule:: _tcn_stream_merger
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TcnStreamMergerOp,
             PyTcnStreamMergerOp,
             holoscan::Operator,
             std::shared_ptr<TcnStreamMergerOp>>(
      m,
      "TcnStreamMergerOp",
      doc::TcnStreamMergerOp::doc_TcnStreamMergerOp)
      .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                    const py::args&,
                    const std::vector<std::string>&,
                    const std::string&,
                    const std::string&,
                    bool,
                    std::shared_ptr<holoscan::Allocator>,
                    const std::string&>(),
           "fragment"_a,
           "input_port_names"_a,
           "input_message_name"_a,
           "output_message_name"_a,
           "fuse_buffers"_a = false,
           "allocator"_a = nullptr,
           "name"_a = "tcn_stream_merger"s,
           doc::TcnStreamMergerOp::doc_TcnStreamMergerOp)
      .def("initialize",
           &TcnStreamMergerOp::initialize,
           doc::TcnStreamMergerOp::doc_initialize)
      .def("setup",
           &TcnStreamMergerOp::setup,
           "spec"_a,
           doc::TcnStreamMergerOp::doc_setup);

  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    HOLOSCAN_LOG_DEBUG("TCN Stream Merger - register types");
  });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
