/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>

#include "macros.hpp"

namespace tcn::doc::TcnStreamSynchronizerOp {

PYDOC(TcnStreamSynchronizerOp, R"doc(
Groups entities arriving on several ports at different rates into frame-groups that share an
acquisition timestamp.

One input port and one identically-named output port are created per entry in `streams`. Each
stream is buffered independently; when every required stream holds an entity whose timestamp
matches under the configured policy, that group is emitted -- one member per output port, each
stamped with the group's timestamp. Payloads are forwarded as whole entities and never inspected,
so any tensor layout works.

Of all complete groups currently buffered the OLDEST is published, and everything strictly older
than it is then discarded in every buffer. Discarding relative to the published group -- rather
than to each stream's own head -- is what stops a lagging stream from having its future partners
thrown away before it catches up.

Requires that upstream stamps its messages. `tcn_shm_subscriber` attaches a `nvidia::gxf::Timestamp`
and every Python stage on the mask path forwards it via `emit(..., acq_timestamp=...)`; a stream
whose messages carry no timestamp cannot be grouped, and the operator reports that rather than
guessing. Timestamps must also advance: a looping replay source that restarts at its first frame
has those messages rejected as non-monotonic unless it offsets each pass.

`streams` MUST be passed to the constructor, not via `from_config()`: ports are created in
`setup()`, which Holoscan runs before parameter values are applied.

Parameters
----------
fragment : holoscan.core.Fragment (constructor positional only)
    The fragment (or subgraph) that the operator belongs to.
streams : list of str
    Stream names, one input and one output port each. Order matters: it indexes `capacities`.
optional_streams : list of str, optional
    Streams that need not be present for a group to be complete. Every entry must also appear in
    `streams`. Default is no optional streams.
capacities : list of int, optional
    Per-stream buffer capacity, in the same order as `streams`. Empty means use `default_capacity`
    for all; otherwise the length must equal `len(streams)`. Size each stream to exceed its
    worst-case skew relative to the slowest stream, so a fast stream needs more slots than a slow
    one.
reference_stream : str, optional
    The stream whose buffer is searched oldest-first for candidate groups. Use the laggiest stream.
    Defaults to the first entry in `streams`, and may not name an optional stream.
match_policy : str, optional
    "exact" requires identical timestamps; "window" accepts any partner within `window_ns`, taking
    the nearest and breaking ties toward the older. Default is "exact", which is correct whenever
    all streams originate from one timestamped source buffer.
window_ns : int, optional
    Tolerance in nanoseconds for `match_policy="window"`; must be > 0 for that policy. Unused by
    "exact". Default is 0.
default_capacity : int, optional
    Buffer capacity for streams not covered by `capacities`. Default is 8.
verbose : bool, optional
    Log every published group and every discard. Default is False; per-stream counters are reported
    at shutdown either way.
name : str, optional
    The name of the operator. Default is "tcn_stream_synchronizer".
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Creates one input and one output port per configured stream. The input ports carry no per-port
condition; instead a single `MultiMessageAvailableCondition` in `kSumOfAll` mode with `min_sum=1`
makes the operator tick when ANY port has data, because waiting for every port at once is exactly
what a rate-mismatched graph can never satisfy.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace tcn::doc::TcnStreamSynchronizerOp
