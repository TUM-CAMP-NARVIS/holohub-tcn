/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace doc {

namespace TcnZenohSubscriberOp {

constexpr const char* doc_TcnZenohSubscriberOp = R"doc(
Holoscan source operator that subscribes to a Zenoh topic.

Receives CDR-encoded samples from Zenoh, queues them, and emits raw
payload bytes on the "output" port. The CDR type name is attached as
metadata ("CdrTypeName") for downstream decoders.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
topic : str
    Zenoh key expression to subscribe to.
async_condition : holoscan.conditions.AsynchronousCondition
    AsynchronousCondition for scheduling.
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

}  // namespace TcnZenohSubscriberOp

}  // namespace doc
