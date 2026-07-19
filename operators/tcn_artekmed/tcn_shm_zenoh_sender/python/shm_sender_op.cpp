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

#include "../shm_sender_op.hpp"
#include "./shm_sender_op_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnShmZenohSenderOp : public TcnShmZenohSenderOp {
 public:
    using TcnShmZenohSenderOp::TcnShmZenohSenderOp;

    PyTcnShmZenohSenderOp(
        const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
        const py::args& args,
        const std::string& stream_name = "camera_streams",
        const std::vector<std::string>& input_tensor_names = {},
        const std::string& name = "tcn_shm_zenoh_sender")
        : TcnShmZenohSenderOp(
              holoscan::ArgList{
                  holoscan::Arg{"stream_name", stream_name},
                  holoscan::Arg{"input_tensor_names", input_tensor_names}}) {
        add_positional_condition_and_resource_args(this, args);
        init_operator_base(this, fragment_or_subgraph, name);
    }
};

PYBIND11_MODULE(_tcn_shm_zenoh_sender, m) {
    m.doc() = R"pbdoc(
        Holoscan SDK TCN SHM Zenoh Sender Python Bindings
        --------------------------------------------------
        .. currentmodule:: _tcn_shm_zenoh_sender
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<TcnShmZenohSenderOp,
               PyTcnShmZenohSenderOp,
               holoscan::Operator,
               std::shared_ptr<TcnShmZenohSenderOp>>(
        m,
        "TcnShmZenohSenderOp",
        doc::TcnShmZenohSenderOp::doc_TcnShmZenohSenderOp)
        .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                      const py::args&,
                      const std::string&,
                      const std::vector<std::string>&,
                      const std::string&>(),
             "fragment"_a,
             "stream_name"_a = "camera_streams"s,
             "input_tensor_names"_a = std::vector<std::string>{},
             "name"_a = "tcn_shm_zenoh_sender"s,
             doc::TcnShmZenohSenderOp::doc_TcnShmZenohSenderOp)
        .def("initialize",
             &TcnShmZenohSenderOp::initialize,
             doc::TcnShmZenohSenderOp::doc_initialize)
        .def("setup",
             &TcnShmZenohSenderOp::setup,
             "spec"_a,
             doc::TcnShmZenohSenderOp::doc_setup);

    m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
        HOLOSCAN_LOG_DEBUG("TCN SHM Zenoh Sender - register types");
    });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
