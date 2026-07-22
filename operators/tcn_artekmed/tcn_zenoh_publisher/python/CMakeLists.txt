# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)

pybind11_add_holohub_module(
    CPP_CMAKE_TARGET tcn_zenoh_publisher
    CLASS_NAME "TcnZenohPublisherOp"
    SOURCES zenoh_publisher_op.cpp
)

# The pybind module recompiles zenoh_publisher_op.cpp, whose header holds a
# std::unique_ptr<zenoh::Publisher> member. zenoh.hxx only fully defines
# zenoh::Publisher when ZENOHCXX_ZENOHC is set, so without the zenoh
# link/definition the module fails with "sizeof incomplete type". Mirror the
# operator library's zenoh settings on the pybind target (as tcn_zenoh_receiver
# already does).
if(TARGET zenohcxx::zenohcxx)
    target_link_libraries(tcn_zenoh_publisher_python PRIVATE zenohcxx::zenohcxx)
elseif(TARGET zenohc::lib)
    target_link_libraries(tcn_zenoh_publisher_python PRIVATE zenohc::lib)
endif()
target_compile_definitions(tcn_zenoh_publisher_python PRIVATE ZENOHCXX_ZENOHC=1)
