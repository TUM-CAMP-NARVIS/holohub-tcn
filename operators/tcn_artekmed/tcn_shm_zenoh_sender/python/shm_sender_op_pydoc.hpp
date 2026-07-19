/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace tcn::ops::doc {

namespace TcnShmZenohSenderOp {

static const char* const doc_TcnShmZenohSenderOp = R"doc(
Holoscan operator that publishes decoded video frames to iceoryx2 SHM.

Takes decoded frame tensors from the Holoscan dataflow and publishes them
to iceoryx2 shared memory for consumption by other local processes via
the {stream_name}/COMPOSITE_BUFFER/Frame service.

Parameters
----------
fragment : holoscan.core.Fragment
    The Fragment the operator belongs to.
stream_name : str, optional
    SHM service name prefix (default: "camera_streams").
input_tensor_names : list[str], optional
    List of tensor names to publish from the input entity.
name : str, optional
    The name of the operator (default: "tcn_shm_zenoh_sender").
)doc";

static const char* const doc_initialize = R"doc(
Initialize the operator.
)doc";

static const char* const doc_setup = R"doc(
Define the operator specification.
)doc";

}  // namespace TcnShmZenohSenderOp

}  // namespace tcn::ops::doc
