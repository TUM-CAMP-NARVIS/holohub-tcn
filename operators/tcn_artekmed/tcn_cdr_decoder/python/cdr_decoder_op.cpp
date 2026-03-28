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

#include "../cdr_decoder_op.hpp"
#include "./cdr_decoder_op_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnCdrDecoderOp : public TcnCdrDecoderOp {
 public:
    using TcnCdrDecoderOp::TcnCdrDecoderOp;

    PyTcnCdrDecoderOp(
        const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
        const py::args& args,
        const std::string& source_name = "",
        int32_t stream_index = 0,
        const std::string& name = "tcn_cdr_decoder")
        : TcnCdrDecoderOp(
              holoscan::ArgList{
                  holoscan::Arg{"source_name", source_name},
                  holoscan::Arg{"stream_index", stream_index}}) {
        add_positional_condition_and_resource_args(this, args);
        init_operator_base(this, fragment_or_subgraph, name);
    }
};

PYBIND11_MODULE(_tcn_cdr_decoder, m) {
    m.doc() = R"pbdoc(
        Holoscan SDK TCN CDR Decoder Python Bindings
        ---------------------------------------------
        .. currentmodule:: _tcn_cdr_decoder
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<TcnCdrDecoderOp,
               PyTcnCdrDecoderOp,
               holoscan::Operator,
               std::shared_ptr<TcnCdrDecoderOp>>(
        m,
        "TcnCdrDecoderOp",
        doc::TcnCdrDecoderOp::doc_TcnCdrDecoderOp)
        .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                      const py::args&,
                      const std::string&,
                      int32_t,
                      const std::string&>(),
             "fragment"_a,
             "source_name"_a = ""s,
             "stream_index"_a = 0,
             "name"_a = "tcn_cdr_decoder"s,
             doc::TcnCdrDecoderOp::doc_TcnCdrDecoderOp)
        .def("initialize",
             &TcnCdrDecoderOp::initialize,
             doc::TcnCdrDecoderOp::doc_initialize)
        .def("setup",
             &TcnCdrDecoderOp::setup,
             "spec"_a,
             doc::TcnCdrDecoderOp::doc_setup);

    m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
        HOLOSCAN_LOG_DEBUG("TCN CDR Decoder - register types");
    });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
