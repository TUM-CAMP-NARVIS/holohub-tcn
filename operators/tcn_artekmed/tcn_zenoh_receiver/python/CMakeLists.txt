# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)

pybind11_add_holohub_module(
    CPP_CMAKE_TARGET tcn_zenoh_receiver
    CLASS_NAME "TcnZenohReceiverOp"
    SOURCES zenoh_receiver_op.cpp
)

# Override the auto-generated __init__.py with our custom one that exports
# additional symbols (ZenohSession, ZenohStreamConfig, open_zenoh_session,
# discover_streams) beyond just TcnZenohReceiverOp.
# NOTE: This must run AFTER pybind11_add_holohub_module's configure_file so
# that our custom __init__.py wins. CMake processes configure_file calls
# sequentially within the same CMakeLists.txt, so ordering is guaranteed here.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/tcn_zenoh_receiver/__init__.py
    COPYONLY
)

# The pybind11 bindings include zenoh.hxx directly (for ZenohSessionWrapper,
# open_zenoh_session, discover_streams) so we need zenoh link/include.
if(TARGET zenohcxx::zenohcxx)
    target_link_libraries(tcn_zenoh_receiver_python PRIVATE zenohcxx::zenohcxx)
elseif(TARGET zenohc::lib)
    target_link_libraries(tcn_zenoh_receiver_python PRIVATE zenohc::lib)
endif()
target_compile_definitions(tcn_zenoh_receiver_python PRIVATE ZENOHCXX_ZENOHC=1)
