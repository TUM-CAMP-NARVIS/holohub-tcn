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

#include "../device_context_service.hpp"
#include "../xy_lookup_table_source_op.hpp"
#include "./device_context_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyXYLookupTableSourceOp : public XYLookupTableSourceOp {
 public:
    using XYLookupTableSourceOp::XYLookupTableSourceOp;

    PyXYLookupTableSourceOp(
        const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
        const py::args& args,
        std::shared_ptr<holoscan::Allocator> allocator,
        const std::string& camera_name = "",
        const std::string& name = "xy_lookup_table_source")
        : XYLookupTableSourceOp(
              holoscan::ArgList{
                  holoscan::Arg{"allocator", allocator},
                  holoscan::Arg{"camera_name", camera_name}}) {
        add_positional_condition_and_resource_args(this, args);
        init_operator_base(this, fragment_or_subgraph, name);
    }
};

PYBIND11_MODULE(_tcn_device_context, m) {
    m.doc() = R"pbdoc(
        Holoscan SDK TCN Device Context Python Bindings
        ------------------------------------------------
        .. currentmodule:: _tcn_device_context
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // DeviceContextService binding
    py::class_<DeviceContextService,
               std::shared_ptr<DeviceContextService>>(
        m,
        "DeviceContextService",
        doc::DeviceContextService::doc_DeviceContextService)
        .def(py::init<>())
        .def("has_camera",
             &DeviceContextService::has_camera,
             "camera_name"_a,
             doc::DeviceContextService::doc_has_camera)
        .def("camera_names",
             &DeviceContextService::camera_names,
             doc::DeviceContextService::doc_camera_names);

    // XYLookupTableSourceOp binding
    py::class_<XYLookupTableSourceOp,
               PyXYLookupTableSourceOp,
               holoscan::Operator,
               std::shared_ptr<XYLookupTableSourceOp>>(
        m,
        "XYLookupTableSourceOp",
        doc::XYLookupTableSourceOp::doc_XYLookupTableSourceOp)
        .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                      const py::args&,
                      std::shared_ptr<holoscan::Allocator>,
                      const std::string&,
                      const std::string&>(),
             "fragment"_a,
             "allocator"_a,
             "camera_name"_a = ""s,
             "name"_a = "xy_lookup_table_source"s,
             doc::XYLookupTableSourceOp::doc_XYLookupTableSourceOp)
        .def("initialize",
             &XYLookupTableSourceOp::initialize,
             doc::XYLookupTableSourceOp::doc_initialize)
        .def("setup",
             &XYLookupTableSourceOp::setup,
             "spec"_a,
             doc::XYLookupTableSourceOp::doc_setup)
        .def("set_device_context_service",
             &XYLookupTableSourceOp::set_device_context_service,
             "service"_a,
             doc::XYLookupTableSourceOp::doc_set_device_context_service);

    m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
        HOLOSCAN_LOG_DEBUG("TCN Device Context - register types");
    });
}  // PYBIND11_MODULE NOLINT

}  // namespace tcn::ops
