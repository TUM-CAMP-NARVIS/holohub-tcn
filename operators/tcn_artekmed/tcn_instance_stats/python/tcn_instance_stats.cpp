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
#include "../tcn_instance_stats.cuh"
#include "./tcn_instance_stats_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

class PyTcnInstanceStatsOp : public TcnInstanceStatsOp {
 public:
  using TcnInstanceStatsOp::TcnInstanceStatsOp;

  PyTcnInstanceStatsOp(
      const std::variant<holoscan::Fragment*, holoscan::Subgraph*>& fragment_or_subgraph,
      const py::args& args,
      std::shared_ptr<holoscan::Allocator> allocator,
      int cuda_device_ordinal = 0,
      const std::string& in_positions_tensor_name = "",
      const std::string& in_labels_tensor_name = "",
      const std::string& out_rows_tensor_name = "rows",
      const std::string& out_labels_tensor_name = "labels",
      int64_t camera_index = 0,
      double trim_percentile = 0.02,
      double trim_margin = 0.05,
      double min_range_m = 0.01,
      int64_t up_axis = 1,
      double min_anisotropy = 1.5,
      int64_t min_points = 64,
      int64_t max_instances = 64,
      bool component_filter = false,
      double component_max_gap_m = 0.05,
      double component_min_fraction = 0.1,
      bool verbose = false,
      const std::string& name = "tcn_instance_stats")
      : TcnInstanceStatsOp(
            holoscan::ArgList{holoscan::Arg{"allocator", allocator},
                              holoscan::Arg{"cuda_device_ordinal", cuda_device_ordinal},
                              holoscan::Arg{"in_positions_tensor_name", in_positions_tensor_name},
                              holoscan::Arg{"in_labels_tensor_name", in_labels_tensor_name},
                              holoscan::Arg{"out_rows_tensor_name", out_rows_tensor_name},
                              holoscan::Arg{"out_labels_tensor_name", out_labels_tensor_name},
                              holoscan::Arg{"camera_index", camera_index},
                              holoscan::Arg{"trim_percentile", trim_percentile},
                              holoscan::Arg{"trim_margin", trim_margin},
                              holoscan::Arg{"min_range_m", min_range_m},
                              holoscan::Arg{"up_axis", up_axis},
                              holoscan::Arg{"min_anisotropy", min_anisotropy},
                              holoscan::Arg{"min_points", min_points},
                              holoscan::Arg{"max_instances", max_instances},
                              holoscan::Arg{"component_filter", component_filter},
                              holoscan::Arg{"component_max_gap_m", component_max_gap_m},
                              holoscan::Arg{"component_min_fraction", component_min_fraction},
                              holoscan::Arg{"verbose", verbose}}) {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

PYBIND11_MODULE(_tcn_instance_stats, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK TCN Instance Stats Python Bindings
        ---------------------------------------
        .. currentmodule:: _tcn_instance_stats
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TcnInstanceStatsOp,
             PyTcnInstanceStatsOp,
             holoscan::Operator,
             std::shared_ptr<TcnInstanceStatsOp>>(
      m, "TcnInstanceStatsOp", doc::TcnInstanceStatsOp::doc_TcnInstanceStatsOp)
      .def(py::init<std::variant<holoscan::Fragment*, holoscan::Subgraph*>,
                    const py::args&,
                    std::shared_ptr<holoscan::Allocator>,
                    int,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    int64_t,
                    double,
                    double,
                    double,
                    int64_t,
                    double,
                    int64_t,
                    int64_t,
                    bool,
                    double,
                    double,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "cuda_device_ordinal"_a = 0,
           "in_positions_tensor_name"_a = ""s,
           "in_labels_tensor_name"_a = ""s,
           "out_rows_tensor_name"_a = "rows"s,
           "out_labels_tensor_name"_a = "labels"s,
           "camera_index"_a = static_cast<int64_t>(0),
           "trim_percentile"_a = 0.02,
           "trim_margin"_a = 0.05,
           "min_range_m"_a = 0.01,
           "up_axis"_a = static_cast<int64_t>(1),
           "min_anisotropy"_a = 1.5,
           "min_points"_a = static_cast<int64_t>(64),
           "max_instances"_a = static_cast<int64_t>(64),
           "component_filter"_a = false,
           "component_max_gap_m"_a = 0.05,
           "component_min_fraction"_a = 0.1,
           "verbose"_a = false,
           "name"_a = "tcn_instance_stats"s,
           doc::TcnInstanceStatsOp::doc_TcnInstanceStatsOp)
      .def("initialize", &TcnInstanceStatsOp::initialize, doc::TcnInstanceStatsOp::doc_initialize)
      .def("setup", &TcnInstanceStatsOp::setup, "spec"_a, doc::TcnInstanceStatsOp::doc_setup);

  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    (void)registry;
    HOLOSCAN_LOG_DEBUG("TCN Instance Stats - register types");
  });
}  // PYBIND11_MODULE NOLINT
}  // namespace tcn::ops
