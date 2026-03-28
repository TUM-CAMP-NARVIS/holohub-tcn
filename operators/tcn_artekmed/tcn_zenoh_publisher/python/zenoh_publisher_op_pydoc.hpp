/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace doc {

namespace TcnZenohPublisherOp {

constexpr const char* doc_TcnZenohPublisherOp = R"doc(
Holoscan sink operator that publishes CDR-encoded payloads to Zenoh.

Takes raw bytes on the "input" port and publishes them to a Zenoh topic.
The CDR type name is read from input metadata ("CdrTypeName") and attached
to the Zenoh sample as an attachment.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
topic : str
    Zenoh key expression to publish on.
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

}  // namespace TcnZenohPublisherOp

}  // namespace doc
