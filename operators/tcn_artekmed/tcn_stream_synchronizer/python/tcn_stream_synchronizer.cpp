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

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "../tcn_stream_synchronizer.hpp"
#include "./tcn_stream_synchronizer_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyTcnStreamSynchronizerOp : public TcnStreamSynchronizerOp {
 public:
  /* Inherit the constructors */
  using TcnStreamSynchronizerOp::TcnStreamSynchronizerOp;

  // Define a constructor that fully initializes the object.
  PyTcnStreamSynchronizerOp(holoscan::Fragment* fragment, const py::args& args, int cuda_device_ordinal,
                     std::shared_ptr<::holoscan::Allocator> allocator, int num_streams, bool verbose,
                     const std::string& name = "nv_video_decoder")
      : TcnStreamSynchronizerOp(holoscan::ArgList{holoscan::Arg{"cuda_device_ordinal", cuda_device_ordinal},
                                 holoscan::Arg{"allocator", allocator},
                                 holoscan::Arg{"num_streams", num_streams},
                                 holoscan::Arg{"verbose", verbose}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_tcn_stream_synchronizer, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _tcn_stream_synchronizer
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TcnStreamSynchronizerOp, PyTcnStreamSynchronizerOp, Operator, std::shared_ptr<TcnStreamSynchronizerOp>>(
      m, "TcnStreamSynchronizerOp", doc::TcnStreamSynchronizerOp::doc_TcnStreamSynchronizerOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    int,
                    std::shared_ptr<::holoscan::Allocator>,
                    int,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "cuda_device_ordinal"_a,
           "allocator"_a,
           "num_streams"_a = 1,
           "verbose"_a = false,
           "name"_a = "tcn_stream_synchronizer"s,
           doc::TcnStreamSynchronizerOp::doc_TcnStreamSynchronizerOp)
      .def("initialize", &TcnStreamSynchronizerOp::initialize, doc::TcnStreamSynchronizerOp::doc_initialize)
      .def("setup", &TcnStreamSynchronizerOp::setup, "spec"_a, doc::TcnStreamSynchronizerOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
