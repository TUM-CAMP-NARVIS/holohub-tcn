/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/subgraph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/python/core/component_util.hpp"
#include <holoscan/python/core/emitter_receiver_registry.hpp>

#define ZENOHCXX_ZENOHC 1
#include <zenoh.hxx>

#include "../zenoh_receiver_op.hpp"
#include "./zenoh_receiver_op_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

/// Opaque wrapper around a C++ zenoh::Session for use from Python.
/// Python cannot pass a Python-zenoh session to C++ operators, so this
/// provides a C++-native session that can be shared between discover_streams()
/// and TcnZenohReceiverOp::set_session().
class ZenohSessionWrapper {
 public:
    explicit ZenohSessionWrapper(std::shared_ptr<zenoh::Session> session)
        : session_(std::move(session)) {}

    std::shared_ptr<zenoh::Session> session() const { return session_; }

 private:
    std::shared_ptr<zenoh::Session> session_;
};

class PyTcnZenohReceiverOp : public TcnZenohReceiverOp {
 public:
    using TcnZenohReceiverOp::TcnZenohReceiverOp;

    PyTcnZenohReceiverOp(
        const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
        const py::args& args,
        std::shared_ptr<holoscan::AsynchronousCondition> async_condition,
        std::shared_ptr<holoscan::Allocator> allocator = nullptr,
        std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
        const std::string& name = "tcn_zenoh_receiver")
        : TcnZenohReceiverOp(
              holoscan::ArgList{
                  holoscan::Arg{"async_condition", async_condition}}) {
        if (allocator) {
            add_arg(holoscan::Arg{"allocator", allocator});
        }
        if (cuda_stream_pool) {
            add_arg(holoscan::Arg{"cuda_stream_pool", cuda_stream_pool});
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

    // Bind ZenohSessionWrapper (opaque C++ zenoh session for Python)
    py::class_<ZenohSessionWrapper, std::shared_ptr<ZenohSessionWrapper>>(
        m, "ZenohSession",
        R"doc(Opaque C++ Zenoh session wrapper.

Use ``open_zenoh_session()`` to create one. Pass it to
``TcnZenohReceiverOp.set_session()`` and ``discover_streams()``.
)doc")
        .def("__repr__", [](const ZenohSessionWrapper&) {
            return "<ZenohSession (C++ zenoh::Session)>";
        });

    // Module-level function: open a C++ zenoh session
    m.def("open_zenoh_session",
        [](const std::string& config_file) -> std::shared_ptr<ZenohSessionWrapper> {
            zenoh::Config config = zenoh::Config::create_default();
            if (!config_file.empty()) {
                config = zenoh::Config::from_file(config_file);
            }
            auto session = zenoh::Session::open(std::move(config));
            auto session_ptr = std::make_shared<zenoh::Session>(std::move(session));
            return std::make_shared<ZenohSessionWrapper>(std::move(session_ptr));
        },
        "config_file"_a = "",
        R"doc(Open a C++ Zenoh session from a config file.

Parameters
----------
config_file : str, optional
    Path to a Zenoh JSON5 config file. If empty, uses default config.

Returns
-------
ZenohSession
    An opaque session handle to pass to ``discover_streams()`` and
    ``TcnZenohReceiverOp.set_session()``.
)doc");

    // Module-level function: discover streams via Zenoh RPC
    m.def("discover_streams",
        [](const std::shared_ptr<ZenohSessionWrapper>& session_wrapper,
           const std::string& topic_prefix,
           const std::string& capture_node,
           const std::vector<std::string>& stream_types)
            -> std::vector<ZenohStreamConfig> {
            if (!session_wrapper || !session_wrapper->session()) {
                throw std::runtime_error("ZenohSession is null");
            }
            return TcnZenohReceiverOp::discover_streams(
                *session_wrapper->session(), topic_prefix, capture_node, stream_types);
        },
        "session"_a,
        "topic_prefix"_a,
        "capture_node"_a,
        "stream_types"_a = std::vector<std::string>{"color", "depth"},
        R"doc(Discover camera streams via Zenoh RPC.

Uses the same two-phase protocol as the C++ application:
  1. GET sensor descriptions from {topic_prefix}/{capture_node}/rpc/sensor/*/describe
  2. Fetch stream descriptors for each enabled sensor/type

Parameters
----------
session : ZenohSession
    An open C++ Zenoh session (from ``open_zenoh_session()``).
topic_prefix : str
    Zenoh topic prefix (e.g. "tcn/loc/pcpd").
capture_node : str
    Capture node name (e.g. "k4a_capture_multi").
stream_types : list of str, optional
    Stream types to discover (default: ["color", "depth"]).

Returns
-------
list of ZenohStreamConfig
    Discovered stream configurations ready for ``set_stream_configs()``.
)doc");

    // Bind ZenohStreamConfig
    py::class_<ZenohStreamConfig>(m, "ZenohStreamConfig")
        .def(py::init<>())
        .def_readwrite("name", &ZenohStreamConfig::name)
        .def_readwrite("topic", &ZenohStreamConfig::topic)
        .def_readwrite("sensor_name", &ZenohStreamConfig::sensor_name)
        .def_readwrite("stream_index", &ZenohStreamConfig::stream_index)
        .def_readwrite("image_width", &ZenohStreamConfig::image_width)
        .def_readwrite("image_height", &ZenohStreamConfig::image_height)
        .def_readwrite("image_step", &ZenohStreamConfig::image_step)
        .def_readwrite("image_format", &ZenohStreamConfig::image_format)
        .def_readwrite("image_compression", &ZenohStreamConfig::image_compression)
        .def_readwrite("frame_rate", &ZenohStreamConfig::frame_rate)
        .def("__repr__", [](const ZenohStreamConfig& cfg) {
            return "<ZenohStreamConfig name='" + cfg.name +
                   "' topic='" + cfg.topic +
                   "' " + std::to_string(cfg.image_width) + "x" +
                   std::to_string(cfg.image_height) +
                   " compression=" + std::to_string(cfg.image_compression) + ">";
        });

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
                      std::shared_ptr<holoscan::CudaStreamPool>,
                      const std::string&>(),
             "fragment"_a,
             "async_condition"_a,
             "allocator"_a = nullptr,
             "cuda_stream_pool"_a = nullptr,
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
             "configs"_a,
             R"doc(Set discovered stream configurations (call before init_spec).

Parameters
----------
configs : list of ZenohStreamConfig
    Stream configurations from ``discover_streams()``.
)doc")
        .def("set_session",
             [](TcnZenohReceiverOp& self,
                const std::shared_ptr<ZenohSessionWrapper>& wrapper) {
                 if (!wrapper || !wrapper->session()) {
                     throw std::runtime_error("ZenohSession is null");
                 }
                 self.set_session(wrapper->session());
             },
             "session"_a,
             R"doc(Set the Zenoh session (call before start).

Parameters
----------
session : ZenohSession
    An open C++ Zenoh session (from ``open_zenoh_session()``).
)doc")
        .def("stream_configs",
             &TcnZenohReceiverOp::stream_configs,
             py::return_value_policy::reference_internal)
        .def("init_spec", &TcnZenohReceiverOp::init_spec);

    m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
        HOLOSCAN_LOG_DEBUG("TCN Zenoh Receiver - register types");
    });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
