// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cdr_serde.hpp"

// This compilation unit ensures the shared library has at least one
// object file.  The template methods in cdr_serde.hpp are header-only
// and instantiated in each consumer.

namespace tcn::cdr {

// Explicit instantiation anchor (prevents empty .so warnings).
// The actual work is done by the header-only templates.

}  // namespace tcn::cdr
