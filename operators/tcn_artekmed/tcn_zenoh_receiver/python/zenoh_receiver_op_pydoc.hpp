/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace doc {

namespace TcnZenohReceiverOp {

constexpr const char* doc_TcnZenohReceiverOp = R"doc(
Composite Holoscan source operator: Zenoh subscription + CDR decode + GPU output.

Subscribes to multiple Zenoh video streams, CDR-decodes payloads using the
CdrTypeRegistry, and emits decoded frames as GPU tensors on dynamic output ports.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
async_condition : holoscan.conditions.AsynchronousCondition
    AsynchronousCondition for event-driven scheduling.
allocator : holoscan.resources.Allocator, optional
    GPU memory allocator for tensor uploads.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    Pool for CUDA streams used in async GPU operations.
name : str, optional
    The name of the operator.

After construction, call ``set_stream_configs()``, ``set_session()``, and
``init_spec()`` before the Holoscan runtime starts.
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

}  // namespace TcnZenohReceiverOp

}  // namespace doc
