/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace doc {

namespace TcnCdrDecoderOp {

constexpr const char* doc_TcnCdrDecoderOp = R"doc(
Holoscan operator that decodes CDR-encoded VideoStreamMessage payloads.

Takes raw CDR bytes from a Zenoh subscriber, deserializes the
VideoStreamMessage using FastCDR, and emits the raw image bytes.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
source_name : str, optional
    Stream source identifier for metadata.
stream_index : int, optional
    Stream index for metadata.
name : str, optional
    The name of the operator.
)doc";

constexpr const char* doc_initialize = R"doc(
Initialize the operator.
)doc";

constexpr const char* doc_setup = R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc";

}  // namespace TcnCdrDecoderOp

}  // namespace doc
