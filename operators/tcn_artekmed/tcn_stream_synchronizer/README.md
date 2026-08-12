# tcn_stream_synchronizer

`TcnStreamSynchronizerOp` — groups entities arriving on several ports at different rates into
frame-groups that share an acquisition timestamp.

Design: [`applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-11-temporal-sync-design.md`](../../../applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-11-temporal-sync-design.md)

## Why

The mask path runs at roughly a third of the source rate and a couple of frames behind it, while the
depth path is independent and faster. Applying a mask to whatever depth frame happens to be current
mis-registers it by a varying amount. This operator buffers each stream and emits only complete,
timestamp-consistent groups.

## Ports

One input **and** one identically-named output per entry in `streams`. Payloads are forwarded as
whole entities and never inspected, so any tensor layout works.

The input ports carry no per-port condition. Instead a single `MultiMessageAvailableCondition` in
`kSumOfAll` mode with `min_sum=1` makes the operator tick when **any** port has data — waiting for
every port at once is exactly what a rate-mismatched graph can never satisfy.

## Parameters

| parameter | default | notes |
|---|---|---|
| `streams` | — | Stream names, one input+output port each. **Must be a constructor argument** (ports are created in `setup()`, before parameters are applied). Order indexes `capacities`. |
| `optional_streams` | `[]` | Streams that need not be present for a group to be complete. Every entry must also appear in `streams`. |
| `capacities` | `[]` | Per-stream buffer capacity, same order as `streams`. Empty means `default_capacity` for all; otherwise the length must equal `len(streams)`. |
| `reference_stream` | first stream | The stream searched oldest-first for candidate groups. Use the laggiest. May not name an optional stream. |
| `match_policy` | `"exact"` | `exact` requires identical timestamps; `window` accepts a partner within `window_ns`, nearest first, ties broken toward the older. |
| `window_ns` | `0` | Tolerance for `match_policy: window`; must be > 0 for that policy. |
| `default_capacity` | `8` | Capacity for streams not covered by `capacities`. |
| `verbose` | `false` | Log every published group and discard. Per-stream counters are reported at shutdown either way. |

Size each stream to exceed its worst-case skew relative to the slowest stream, so a fast stream needs
several times more slots than a slow one.

`match_policy: exact` is correct whenever all cameras arrive through one shared-memory segment: the
subscriber stamps one acquisition time per composite buffer and every camera shares it.

## The rule

Of all complete groups currently buffered, publish the **oldest**, then discard everything strictly
older than it in **every** buffer. Discarding relative to the published group — rather than to each
stream's own head — is what stops a lagging stream from having its future partners thrown away before
it catches up. The group's timestamp is the minimum of its members, so the discard boundary can never
drop a member that was just published.

## Requirements and caveats

- **Upstream must stamp its messages.** A stream whose messages carry no timestamp cannot be grouped;
  the operator reports it rather than guessing. See the collection README on acquisition timestamps.
- **Timestamps must advance.** A timestamp is a frame's identity, so a non-advancing one is rejected
  and counted as `non-monotonic`. A looping replay source that restarts at its first frame therefore
  has passes 2+ rejected unless it offsets them — `tcn_dataset_replayer` does exactly that.
- **Starvation is reported, not silent.** `log_starvation()` names the streams preventing a match,
  rate-limited. If a buffer fills while still unmatched, `force_progress()` drops the fullest stream's
  oldest entry so the graph makes progress instead of wedging, and says so.

## Shutdown report

```
TcnStreamSynchronizerOp: published 8 groups, 0 ticks without a match, 0 forced drops
  masks: received=8 unstamped=0 evicted=0 non-monotonic=0 occupancy=0/4
  depth: received=8 unstamped=0 evicted=0 non-monotonic=0 occupancy=0/16
```

`unstamped` > 0 means a producer is not forwarding timestamps. `non-monotonic` rising against a
looping source means that source replays raw timestamps. `forced drops` > 0 means a capacity is too
small for the actual skew.

## Usage

```python
from holohub.tcn_stream_synchronizer import TcnStreamSynchronizerOp

sync = TcnStreamSynchronizerOp(
    self,
    streams=["masks", "depth"],          # constructor argument, not from_config()
    capacities=[4, 16],
    reference_stream="masks",            # the laggiest stream drives the search
    match_policy="exact",
    name="temporal_sync")
self.add_flow(source, sync, {("depth_outputs", "depth")})
self.add_flow(mask_path, sync, {("output_masks", "masks")})
for s in ("masks", "depth"):
    self.add_flow(sync, consumer, {(s, s)})
```

## Internals and tests

- `ring_buffer.hpp` — `RingBuffer<T>` of `Timed<T>`, deque-backed with enforced capacity so eviction
  frees the payload by construction. Rejects non-advancing timestamps and counts them; monotonicity
  spans the buffer's lifetime, so it survives eviction.
- `frame_matcher.hpp` — `MatchPolicy` (`ExactMatch`, `WindowMatch`) and `find_oldest_group()`.
- `tests/test_sync_logic.cpp` — self-contained, no gtest:

  ```bash
  g++ -std=c++17 -O1 -o t tests/test_sync_logic.cpp && ./t     # 62 checks
  ```
