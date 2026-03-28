/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "../zenoh_receiver_op.hpp"
#include "./zenoh_receiver_op_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnZenohReceiverOp : public TcnZenohReceiverOp {
 public:
    using TcnZenohReceiverOp::TcnZenohReceiverOp;

    PyTcnZenohReceiverOp(
        const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
        const py::args& args,
        std::shared_ptr<holoscan::AsynchronousCondition> async_condition,
        std::shared_ptr<holoscan::Allocator> allocator = nullptr,
        const std::string& name = "tcn_zenoh_receiver")
        : TcnZenohReceiverOp(
              holoscan::ArgList{
                  holoscan::Arg{"async_condition", async_condition}}) {
        if (allocator) {
            add_arg(holoscan::Arg{"allocator", allocator});
        }
        add_positional_condition_and_resource_args(this, args);
        init_operator_base(this, fragment_or_subgraph, name);
    }
};

PYBIND11_MODULE(_tcn_zenoh_receiver, m) {
    m.doc() = R"pbdoc(
        Holoscan SDK TCN Zenoh Receiver Python Bindings
        ------------------------------------------------
        .. currentmodule:: _tcn_zenoh_receiver
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // Bind ZenohStreamConfig
    py::class_<ZenohStreamConfig>(m, "ZenohStreamConfig")
        .def(py::init<>())
        .def_readwrite("name", &ZenohStreamConfig::name)
        .def_readwrite("topic", &ZenohStreamConfig::topic)
        .def_readwrite("sensor_name", &ZenohStreamConfig::sensor_name)
        .def_readwrite("stream_index", &ZenohStreamConfig::stream_index)
        .def_readwrite("image_width", &ZenohStreamConfig::image_width)
        .def_readwrite("image_height", &ZenohStreamConfig::image_height)
        .def_readwrite("image_format", &ZenohStreamConfig::image_format)
        .def_readwrite("image_compression", &ZenohStreamConfig::image_compression)
        .def_readwrite("frame_rate", &ZenohStreamConfig::frame_rate);

    py::class_<TcnZenohReceiverOp,
               PyTcnZenohReceiverOp,
               holoscan::Operator,
               std::shared_ptr<TcnZenohReceiverOp>>(
        m,
        "TcnZenohReceiverOp",
        doc::TcnZenohReceiverOp::doc_TcnZenohReceiverOp)
        .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                      const py::args&,
                      std::shared_ptr<holoscan::AsynchronousCondition>,
                      std::shared_ptr<holoscan::Allocator>,
                      const std::string&>(),
             "fragment"_a,
             "async_condition"_a,
             "allocator"_a = nullptr,
             "name"_a = "tcn_zenoh_receiver"s,
             doc::TcnZenohReceiverOp::doc_TcnZenohReceiverOp)
        .def("initialize",
             &TcnZenohReceiverOp::initialize,
             doc::TcnZenohReceiverOp::doc_initialize)
        .def("setup",
             &TcnZenohReceiverOp::setup,
             "spec"_a,
             doc::TcnZenohReceiverOp::doc_setup)
        .def("set_stream_configs",
             &TcnZenohReceiverOp::set_stream_configs,
             "configs"_a)
        .def("stream_configs",
             &TcnZenohReceiverOp::stream_configs,
             py::return_value_policy::reference_internal)
        .def("init_spec", &TcnZenohReceiverOp::init_spec);

    m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
        HOLOSCAN_LOG_DEBUG("TCN Zenoh Receiver - register types");
    });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
