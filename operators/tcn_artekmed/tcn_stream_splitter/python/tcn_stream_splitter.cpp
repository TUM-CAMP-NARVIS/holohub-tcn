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

#include "../tcn_stream_splitter.hpp"
#include "./tcn_stream_splitter_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnStreamSplitterOp : public TcnStreamSplitterOp {
 public:
  using TcnStreamSplitterOp::TcnStreamSplitterOp;

  PyTcnStreamSplitterOp(
      const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
      const py::args& args,
      const std::vector<std::string>& channel_names,
      const std::string& name = "tcn_stream_splitter")
      : TcnStreamSplitterOp(
            holoscan::ArgList{holoscan::Arg{"channel_names", channel_names}}) {
    // Store port names before init_operator_base triggers setup()
    channel_names_init_ = channel_names;
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

PYBIND11_MODULE(_tcn_stream_splitter, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK TCN Stream Splitter Python Bindings
        ---------------------------------------
        .. currentmodule:: _tcn_stream_splitter
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TcnStreamSplitterOp,
             PyTcnStreamSplitterOp,
             holoscan::Operator,
             std::shared_ptr<TcnStreamSplitterOp>>(
      m,
      "TcnStreamSplitterOp",
      doc::TcnStreamSplitterOp::doc_TcnStreamSplitterOp)
      .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                    const py::args&,
                    const std::vector<std::string>&,
                    const std::string&>(),
           "fragment"_a,
           "channel_names"_a,
           "name"_a = "tcn_stream_splitter"s,
           doc::TcnStreamSplitterOp::doc_TcnStreamSplitterOp)
      .def("initialize",
           &TcnStreamSplitterOp::initialize,
           doc::TcnStreamSplitterOp::doc_initialize)
      .def("setup",
           &TcnStreamSplitterOp::setup,
           "spec"_a,
           doc::TcnStreamSplitterOp::doc_setup);

  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    HOLOSCAN_LOG_DEBUG("TCN Stream Splitter - register types");
  });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
