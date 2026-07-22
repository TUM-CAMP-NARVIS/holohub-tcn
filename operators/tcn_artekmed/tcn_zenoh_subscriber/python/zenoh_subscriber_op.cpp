/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <variant>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/subgraph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/python/core/component_util.hpp"
#include <holoscan/python/core/emitter_receiver_registry.hpp>

// Pull in the full zenoh definitions before the operator header: the operator
// holds a std::unique_ptr<zenoh::Subscriber<void>> whose (implicit) destructor
// is instantiated in this translation unit, so zenoh::Subscriber must be a
// complete type here. Mirrors tcn_zenoh_receiver's python binding.
#include <zenoh.hxx>

#include "../zenoh_subscriber_op.hpp"
#include "./zenoh_subscriber_op_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnZenohSubscriberOp : public TcnZenohSubscriberOp {
 public:
    using TcnZenohSubscriberOp::TcnZenohSubscriberOp;

    PyTcnZenohSubscriberOp(
        const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
        const py::args& args,
        std::shared_ptr<holoscan::AsynchronousCondition> async_condition,
        const std::string& topic = "",
        const std::string& name = "tcn_zenoh_subscriber")
        : TcnZenohSubscriberOp(
              holoscan::ArgList{
                  holoscan::Arg{"async_condition", async_condition},
                  holoscan::Arg{"topic", topic}}) {
        add_positional_condition_and_resource_args(this, args);
        init_operator_base(this, fragment_or_subgraph, name);
    }
};

PYBIND11_MODULE(_tcn_zenoh_subscriber, m) {
    m.doc() = R"pbdoc(
        Holoscan SDK TCN Zenoh Subscriber Python Bindings
        -------------------------------------------------
        .. currentmodule:: _tcn_zenoh_subscriber
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<TcnZenohSubscriberOp,
               PyTcnZenohSubscriberOp,
               holoscan::Operator,
               std::shared_ptr<TcnZenohSubscriberOp>>(
        m,
        "TcnZenohSubscriberOp",
        doc::TcnZenohSubscriberOp::doc_TcnZenohSubscriberOp)
        .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                      const py::args&,
                      std::shared_ptr<holoscan::AsynchronousCondition>,
                      const std::string&,
                      const std::string&>(),
             "fragment"_a,
             "async_condition"_a,
             "topic"_a = ""s,
             "name"_a = "tcn_zenoh_subscriber"s,
             doc::TcnZenohSubscriberOp::doc_TcnZenohSubscriberOp)
        .def("initialize",
             &TcnZenohSubscriberOp::initialize,
             doc::TcnZenohSubscriberOp::doc_initialize)
        .def("setup",
             &TcnZenohSubscriberOp::setup,
             "spec"_a,
             doc::TcnZenohSubscriberOp::doc_setup);

    m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
        HOLOSCAN_LOG_DEBUG("TCN Zenoh Subscriber - register types");
    });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
