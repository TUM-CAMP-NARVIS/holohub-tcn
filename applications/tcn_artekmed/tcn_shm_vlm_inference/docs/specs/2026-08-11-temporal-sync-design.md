# Temporal synchronisation of streams (design)

A C++ operator that groups entities arriving on several input ports at different rates into
frame-groups sharing an acquisition timestamp, so masks can be applied to the depth/point-cloud data
they were actually computed from.

Replaces the existing `operators/tcn_artekmed/tcn_stream_synchronizer`, whose registration is
commented out in that tree's `CMakeLists.txt`. That implementation only does *rendezvous* — it waits
for one message per port (which Holoscan's default `MessageAvailable` conditions already do) and
logs enter/emit times. It has no timestamp matching and an unresolved
`// what happens if no videobuffer is present?`.

## Motivation

The mask path runs at ~5 fps and **~2 frames behind the source** — measured, not assumed: the replay
harness's `index_manifest.tsv` recorded arrival index 0 ↔ source frame 2. The depth path is
independent and faster. Applying a mask to whichever depth frame happens to be current therefore
mis-registers it by a variable amount.

`tcn_depthimage_apply_mask` already exists and takes `depth_image` + `mask_image`. The missing piece
is the join, not the application.

---

## Part 0 — prerequisite: frame identity must exist downstream (blocking)

**Today it does not.** `tcn_shm_subscriber` receives `ShmZeroCopyFrame::timestamp` and only *logs*
it (`shm_subscriber_op.cpp:192`); it never attaches it to the emitted entity. There is nothing
downstream to match on, so this must land first.

### 0.1 The mechanism already exists — use it

`nvidia::gxf::Timestamp` (`gxf/std/timestamp.hpp`) carries `acqtime` and `pubtime`, and **Holoscan
already reads it on receive**:

```cpp
// holoscan/core/io_context.hpp
std::optional<int64_t> get_acquisition_timestamp(const char* input_port_name = nullptr);
std::vector<std::optional<int64_t>> get_acquisition_timestamps(...);
```

with a per-input-port map reset before each `compute`. So only the **write** side is missing: the
subscriber must add a `Timestamp` component with `acqtime = frame.timestamp` to both the
`color_outputs` and `depth_outputs` entities.

**One timestamp per SHM composite buffer, shared by every camera in it.** The cameras arriving
through one shared-memory segment are synchronised at capture, and `ShmPortView` has no per-port
timestamp field — the timestamp is per frame-group by construction. So the matching key is the
**capture group**, which is exactly the right granularity here.

### 0.2 Carrying identity through the Python LangSAM path — verify, do not assume

GXF `Timestamp` does **not** survive that path: Python operators emit fresh dicts, producing a new
entity with no `Timestamp` component. Some explicit carrier is required.

`MetadataDictionary` propagates automatically with a merge policy and is the obvious candidate, but
it is **reported unreliable with multiple input ports** — and `MaskCollectorOp` is exactly that
(`IOSpec.ANY_SIZE` receivers). So metadata must not be load-bearing until proven.

**Step 0.2 is therefore an experiment, not an implementation**, and it gates the rest:

1. enable metadata, set `acq_timestamp` in `GdinoOp`, and check it arrives intact at the far side of
   `MaskCollectorOp` (the multi-input hop) and at `LabelMapColorizeOp`;
2. if it survives — use it, with the policy stated explicitly rather than defaulted;
3. if it does not — fall back to threading the value through the existing payload dicts
   (`GdinoOp` → `SamOp` → `PanopticOp` already pass plain dicts internally) and have the final
   emitted TensorMap carry it under a reserved key.

The fallback has a known consequence to handle rather than discover: consumers that iterate the mask
map — `MaskCollectorOp`, `LabelMapColorizeOp`, `MaskDumpOp` — would otherwise treat the extra key as
a camera and, in `MaskDumpOp`'s case, write a bogus `.npy`. Any reserved key must be filtered at
those three sites.

Whichever wins, the synchroniser reads the timestamp through **one** accessor so the choice stays
isolated.

---

## Part 1 — `EntityRingBuffer`: a timestamped KV store for entities

The foundation. Fixed-capacity, per-stream, holding `(timestamp, entity)` pairs.

```cpp
struct TimedEntity {
  int64_t timestamp = 0;                 // acqtime, ns
  holoscan::gxf::Entity entity;          // handle; released when evicted
};

class EntityRingBuffer {
 public:
  explicit EntityRingBuffer(std::size_t capacity);

  bool   push(int64_t timestamp, holoscan::gxf::Entity&& e);  // MOVES in; evicts oldest when full
  std::size_t size()     const;    // occupied slots
  std::size_t capacity() const;
  bool   empty()         const;

  // iteration in ascending timestamp order, for the matcher
  const_iterator begin() const;
  const_iterator end()   const;

  std::optional<TimedEntity> take(int64_t timestamp);   // MOVES out, frees the slot
  std::size_t discard_older_than(int64_t timestamp);    // strictly older; returns count freed
};
```

Requirements, mapped to the brief:

| requirement | how |
|---|---|
| (a) entities move in and out; handle freed on eviction | `push(&&)` / `take()` move; eviction and `discard_older_than` reset the handle so the underlying GXF entity refcount drops and pooled memory returns |
| (b) owner can iterate every element | `begin()`/`end()`, ascending timestamp |
| (c) each item holds timestamp + entity handle | `TimedEntity` |
| (d) abstract match algorithm, concrete policies | Part 2 |
| (e) configurable size, occupancy queryable | ctor `capacity`, `size()` |

Notes that matter for correctness rather than style:

- **Eviction must actually release.** A ring buffer that overwrites a slot without resetting the
  `Entity` keeps the GXF entity alive, and these hold device tensors from a `BlockMemoryPool` — a
  leak here exhausts the pool and stalls the graph rather than growing RSS. Worth an explicit test.
- **Ascending order is an invariant, not an assumption.** Push is expected monotonic per stream; a
  non-monotonic arrival must be handled deliberately (reject with a counter, not silently inserted
  out of order, or the matcher's ordering guarantees break).
- **Capacity is per stream, not global.** It must exceed that stream's worst-case skew relative to
  the slowest stream. With masks ~5 fps and depth ~30 fps, depth needs roughly 6–10 slots to span
  one mask interval while masks need ~2. A single global `maxN` would either starve depth or waste
  pool memory on masks.

---

## Part 2 — matching

### 2.1 The rule

A **complete group** is a set of timestamps — one entity per *required* stream — satisfying the
match predicate. Of all complete groups currently present, publish the one with the **smallest
(oldest) timestamp**, then discard everything strictly older than it in every buffer.

This ordering is what makes it safe. Publishing oldest-first preserves stream order and bounds
latency, and discarding *relative to the published group* rather than to each stream's own head is
what prevents the mask path — permanently ~2 frames behind — from having its future partners thrown
away before it catches up.

Optional streams (e.g. colour) join the group when a match exists and are simply absent otherwise;
they never block or trigger a discard.

### 2.2 Abstract predicate, concrete policies

```cpp
class MatchPolicy {
 public:
  virtual ~MatchPolicy() = default;
  // Candidate timestamps for `stream` that match `reference`, in preference order.
  virtual bool matches(int64_t reference, int64_t candidate) const = 0;
  virtual const char* name() const = 0;
};
```

- `ExactMatch` — `candidate == reference`. Correct for one SHM segment, where every camera in a
  capture group shares an acqtime by construction. **The default.**
- `WindowMatch(tolerance_ns)` — `|candidate - reference| <= tolerance`. For genuinely independent
  sources. Needs a stated tie-break when several candidates fall inside the window (nearest, then
  oldest) or matching becomes order-dependent.

Keeping this behind an interface is worth it because the exact-match case is the one we can reason
about, and the windowed case is where subtle bugs live.

### 2.3 Search

Streams are few (single digits) and buffers small (single/low double digits), so the straightforward
approach is correct and fast enough: iterate the *reference* stream — the one expected to lag most,
i.e. the masks — oldest first; for each of its timestamps, test every other required stream for a
match; the first fully satisfied timestamp is the group to publish.

Deliberately not optimised. At these sizes the cost is negligible next to a single H2D copy, and a
clever index would be harder to reason about than the thing it replaces.

### 2.4 Starvation, which is the real failure mode

If a required stream never produces a matching timestamp, nothing is ever published and every buffer
fills. That must be observable rather than silent:

- count and log (rate-limited) consecutive `compute` calls that find no complete group;
- when a buffer is full and still unmatched, log which streams are blocking and the timestamp spans
  present, then drop that stream's oldest to make progress. Dropping loses a frame; not dropping
  wedges the graph, so the choice is deliberate and must be visible.
- expose occupancy per stream, so the harness can assert buffers are not creeping toward full.

---

## Part 3 — operator surface

`tcn_temporal_sync`, in `operators/tcn_artekmed/tcn_stream_synchronizer/` (replacing the current
implementation; registration to be re-enabled in that tree's `CMakeLists.txt`).

**One port per semantic group**, not per tensor: `masks`, `depth`, `color`, `pointcloud`. The
matching key is the capture group, and every camera in a group shares its timestamp — so per-camera
ports would multiply port count by camera count while adding nothing. Cameras stay as named tensors
inside one entity, exactly as `tcn_shm_subscriber` already emits them.

```yaml
temporal_sync:
  streams:
    - { name: masks,      required: true,  capacity: 4 }
    - { name: depth,      required: true,  capacity: 12 }
    - { name: color,      required: false, capacity: 12 }
    - { name: pointcloud, required: false, capacity: 12 }
  reference_stream: masks        # searched oldest-first; should be the laggiest
  match_policy: exact           # exact | window
  window_ns: 0
  verbose: false
```

Output: one port per input stream, carrying that stream's member of the published group, each
re-stamped with the group timestamp so downstream consumers see a consistent acqtime.

## Testing

| test | where | gate |
|---|---|---|
| `EntityRingBuffer` unit tests | host, gtest | push/evict/take/discard/occupancy; **eviction releases the entity** (refcount or an instrumented deleter) |
| non-monotonic push | host | rejected and counted, never inserted out of order |
| matcher, synthetic timestamps | host | oldest complete group chosen; strictly-older discarded; optional streams do not block; window tie-break deterministic |
| starvation | host | one stream silent → bounded buffers, blocking streams named, documented drop |
| end-to-end alignment | container, replay harness | with a known ~2-frame mask lag, every published group has one acqtime across all ports — the harness makes this checkable because source frames are deterministic |
| no pool exhaustion | container | run long enough that eviction dominates; `BlockMemoryPool` must not run dry |

The ring buffer and matcher are pure logic and belong in host gtest — they are where off-by-one and
lifetime bugs live, and they need no GPU.

## Risks

- **Entity lifetime.** The dominant risk. These handles own pooled device memory; a missed release
  exhausts the pool and stalls the graph rather than failing loudly.
- **Part 0.2 is unresolved** and gates everything. If neither metadata nor a payload field carries
  the timestamp cleanly through the Python path, the alternative is moving the mask path's tail to
  C++ — much larger, and worth knowing before committing.
- **The mask cadence is about to change.** The temporal-decoupling work
  ([`2026-08-11-temporal-decoupling-design.md`](./2026-08-11-temporal-decoupling-design.md)) would
  make masks arrive on a different schedule. The design should not assume a fixed ratio, which is
  why capacity is configurable per stream and the reference stream is named rather than inferred.
- **Rendezvous vs buffering.** Holoscan's default input conditions fire `compute` only when *every*
  input port has a message, which would defeat buffering. The operator needs its input conditions
  set so it can run when *any* port has data — otherwise it degenerates into the rendezvous
  behaviour of the implementation it replaces.

## Out of scope

- Interpolating or warping masks to a non-matching depth frame. This operator matches or does not.
- Changing `tcn_depthimage_apply_mask`.
- Per-camera (rather than per-group) timestamps — not available from the SHM layer, and not
  meaningful for one synchronised segment.
