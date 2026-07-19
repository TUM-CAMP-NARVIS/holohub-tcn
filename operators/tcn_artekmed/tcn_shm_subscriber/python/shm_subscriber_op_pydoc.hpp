/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace tcn::ops::doc {

namespace ShmSynchronizedBufferReceiver {

static const char* const doc_ShmSynchronizedBufferReceiver = R"doc(
Low-level iceoryx2 subscriber for receiving SHM camera data.

Provides camera device discovery, calibration retrieval, channel configuration,
and blocking frame reception with zero-copy buffer access.
)doc";

static const char* const doc_discover_devices = R"doc(
Discover available camera devices via iceoryx2 service listing.

Returns:
    list[str]: Sorted list of camera device names.
)doc";

}  // namespace ShmSynchronizedBufferReceiver

namespace TcnShmSubscriberOp {

static const char* const doc_TcnShmSubscriberOp = R"doc(
Holoscan operator that subscribes to SHM camera streams via iceoryx2.

Wraps ShmSynchronizedBufferReceiver in a background thread, using an
AsynchronousCondition to wake the Holoscan scheduler when new data arrives.

Outputs color and depth tensors keyed by camera port names.

Parameters
----------
fragment : holoscan.core.Fragment
    The Fragment the operator belongs to.
stream_name : str
    SHM stream name to subscribe to.
cycle_time_ms : int, optional
    Wait time between frame polls in milliseconds (default: 1).
allocator : holoscan.resources.Allocator
    Memory allocator for output tensors.
name : str, optional
    The name of the operator (default: "tcn_shm_subscriber").
)doc";

static const char* const doc_initialize = R"doc(
Initialize the operator.
)doc";

static const char* const doc_setup = R"doc(
Define the operator specification.
)doc";

}  // namespace TcnShmSubscriberOp

}  // namespace tcn::ops::doc
