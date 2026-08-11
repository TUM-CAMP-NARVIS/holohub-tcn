/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
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
 *
 * Rewritten alongside the operator. The previous version did not compile at all -- it used
 * unqualified `Operator`/`Fragment` inside `namespace tcn::ops` while closing with
 * `// namespace holoscan::ops`, and bound the old `num_streams`/`cuda_device_ordinal` parameters.
 * That is consistent with the operator's registration having been commented out.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/subgraph.hpp"
#include "holoscan/python/core/component_util.hpp"
#include <holoscan/python/core/emitter_receiver_registry.hpp>

#include "../tcn_stream_synchronizer.hpp"
#include "./tcn_stream_synchronizer_pydoc.hpp"

#include "../../../operator_util.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace tcn::ops {

class PyTcnStreamSynchronizerOp : public TcnStreamSynchronizerOp {
 public:
  using TcnStreamSynchronizerOp::TcnStreamSynchronizerOp;

  PyTcnStreamSynchronizerOp(
      const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
      const py::args& args,
      const std::vector<std::string>& streams,
      const std::vector<std::string>& optional_streams = {},
      const std::vector<int64_t>& capacities = {},
      const std::string& reference_stream = "",
      const std::string& match_policy = "exact",
      int64_t window_ns = 0,
      int64_t default_capacity = 8,
      bool verbose = false,
      const std::string& name = "tcn_stream_synchronizer")
      : TcnStreamSynchronizerOp(
            holoscan::ArgList{holoscan::Arg{"streams", streams},
                              holoscan::Arg{"optional_streams", optional_streams},
                              holoscan::Arg{"capacities", capacities},
                              holoscan::Arg{"reference_stream", reference_stream},
                              holoscan::Arg{"match_policy", match_policy},
                              holoscan::Arg{"window_ns", window_ns},
                              holoscan::Arg{"default_capacity", default_capacity},
                              holoscan::Arg{"verbose", verbose}}) {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

PYBIND11_MODULE(_tcn_stream_synchronizer, m) {
  m.doc() = R"pbdoc(
        TCN temporal stream synchronizer
        ---------------------------------------
        .. currentmodule:: _tcn_stream_synchronizer
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TcnStreamSynchronizerOp,
             PyTcnStreamSynchronizerOp,
             holoscan::Operator,
             std::shared_ptr<TcnStreamSynchronizerOp>>(
      m,
      "TcnStreamSynchronizerOp",
      doc::TcnStreamSynchronizerOp::doc_TcnStreamSynchronizerOp)
      .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                    const py::args&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<int64_t>&,
                    const std::string&,
                    const std::string&,
                    int64_t,
                    int64_t,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "streams"_a,
           "optional_streams"_a = std::vector<std::string>{},
           "capacities"_a = std::vector<int64_t>{},
           "reference_stream"_a = ""s,
           "match_policy"_a = "exact"s,
           "window_ns"_a = static_cast<int64_t>(0),
           "default_capacity"_a = static_cast<int64_t>(8),
           "verbose"_a = false,
           "name"_a = "tcn_stream_synchronizer"s,
           doc::TcnStreamSynchronizerOp::doc_TcnStreamSynchronizerOp)
      .def("setup",
           &TcnStreamSynchronizerOp::setup,
           "spec"_a,
           doc::TcnStreamSynchronizerOp::doc_setup);

  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    (void)registry;
    HOLOSCAN_LOG_DEBUG("TCN Stream Synchronizer - register types");
  });
}  // PYBIND11_MODULE
}  // namespace tcn::ops
