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

#include "../../common/datatypes.hpp"
#include "../tcn_label_sampler.cuh"
#include "./tcn_label_sampler_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnLabelSamplerOp : public TcnLabelSamplerOp {
 public:
  using TcnLabelSamplerOp::TcnLabelSamplerOp;

  PyTcnLabelSamplerOp(
      const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
      const py::args& args,
      std::shared_ptr<holoscan::Allocator> allocator,
      int cuda_device_ordinal = 0,
      const std::string& in_labels_tensor_name = "",
      const std::string& in_texcoord_tensor_name = "",
      const std::string& out_labels_tensor_name = "",
      const std::string& out_mask_tensor_name = "",
      const std::vector<int64_t>& select_classes = {},
      int64_t unlabeled_value = 0,
      const std::string& name = "tcn_label_sampler")
      : TcnLabelSamplerOp(
            holoscan::ArgList{holoscan::Arg{"allocator", allocator},
                              holoscan::Arg{"cuda_device_ordinal", cuda_device_ordinal},
                              holoscan::Arg{"in_labels_tensor_name", in_labels_tensor_name},
                              holoscan::Arg{"in_texcoord_tensor_name", in_texcoord_tensor_name},
                              holoscan::Arg{"out_labels_tensor_name", out_labels_tensor_name},
                              holoscan::Arg{"out_mask_tensor_name", out_mask_tensor_name},
                              holoscan::Arg{"select_classes", select_classes},
                              holoscan::Arg{"unlabeled_value", unlabeled_value}}) {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

PYBIND11_MODULE(_tcn_label_sampler, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK TCN Label Sampler Python Bindings
        ---------------------------------------
        .. currentmodule:: _tcn_label_sampler
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TcnLabelSamplerOp,
             PyTcnLabelSamplerOp,
             holoscan::Operator,
             std::shared_ptr<TcnLabelSamplerOp>>(
      m, "TcnLabelSamplerOp", doc::TcnLabelSamplerOp::doc_TcnLabelSamplerOp)
      .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                    const py::args&,
                    std::shared_ptr<holoscan::Allocator>,
                    int,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::vector<int64_t>&,
                    int64_t,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "cuda_device_ordinal"_a = 0,
           "in_labels_tensor_name"_a = ""s,
           "in_texcoord_tensor_name"_a = ""s,
           "out_labels_tensor_name"_a = ""s,
           "out_mask_tensor_name"_a = ""s,
           "select_classes"_a = std::vector<int64_t>{},
           "unlabeled_value"_a = static_cast<int64_t>(0),
           "name"_a = "tcn_label_sampler"s,
           doc::TcnLabelSamplerOp::doc_TcnLabelSamplerOp)
      .def("initialize", &TcnLabelSamplerOp::initialize, doc::TcnLabelSamplerOp::doc_initialize)
      .def("setup", &TcnLabelSamplerOp::setup, "spec"_a, doc::TcnLabelSamplerOp::doc_setup);

  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    (void)registry;
    HOLOSCAN_LOG_DEBUG("TCN Label Sampler - register types");
  });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
