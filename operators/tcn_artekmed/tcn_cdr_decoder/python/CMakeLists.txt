# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)

pybind11_add_holohub_module(
    CPP_CMAKE_TARGET tcn_cdr_decoder
    CLASS_NAME "TcnCdrDecoderOp"
    SOURCES cdr_decoder_op.cpp
)
