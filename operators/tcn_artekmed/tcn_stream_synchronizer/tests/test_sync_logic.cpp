/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host tests for the temporal-sync logic. Self-contained on purpose: no gtest (absent on both the
 * host and in the runtime container) and no Holoscan, so this builds and runs anywhere with a C++17
 * compiler:
 *
 *   g++ -std=c++17 -O1 -o test_sync_logic test_sync_logic.cpp && ./test_sync_logic
 *
 * That matters because this is precisely the logic that must be right BEFORE a GPU is involved:
 * payload lifetime, ordering invariants, and the group-selection rule.
 */

#include "../frame_matcher.hpp"
#include "../ring_buffer.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace tcn::sync;

static int  g_failures = 0;
static int  g_checks = 0;

#define CHECK(cond, what)                                                        \
  do {                                                                           \
    ++g_checks;                                                                   \
    if (!(cond)) {                                                                \
      ++g_failures;                                                               \
      std::printf("  FAIL %-58s (%s:%d)\n", (what), __FILE__, __LINE__);          \
    }                                                                             \
  } while (0)

// ---------------------------------------------------------------------------------------------
// A payload that reports its own destruction. This is how "eviction frees the handle" is actually
// verified rather than asserted: in production the payload owns pooled device memory, and a missed
// destruction exhausts the BlockMemoryPool and stalls the graph instead of failing loudly.
// ---------------------------------------------------------------------------------------------
struct Tracked {
  static int live;
  static int destroyed;
  int id = 0;

  Tracked() { ++live; }
  explicit Tracked(int i) : id(i) { ++live; }
  Tracked(Tracked&& o) noexcept : id(o.id) { o.id = -1; ++live; }
  Tracked& operator=(Tracked&& o) noexcept {
    id = o.id; o.id = -1; return *this;
  }
  Tracked(const Tracked&) = delete;
  Tracked& operator=(const Tracked&) = delete;
  ~Tracked() { --live; ++destroyed; }
};
int Tracked::live = 0;
int Tracked::destroyed = 0;

static void reset_tracked() { Tracked::live = 0; Tracked::destroyed = 0; }

// ---------------------------------------------------------------------------------------------
static void test_push_size_capacity() {
  std::printf("push / size / capacity\n");
  RingBuffer<int> b(3);
  CHECK(b.capacity() == 3, "capacity honoured");
  CHECK(b.empty(), "starts empty");
  CHECK(b.push(10, 1), "first push accepted");
  CHECK(b.push(20, 2), "second push accepted");
  CHECK(b.size() == 2, "occupancy reported");
  CHECK(!b.full(), "not full at 2/3");
  CHECK(b.push(30, 3), "third push accepted");
  CHECK(b.full(), "full at 3/3");

  RingBuffer<int> z(0);
  CHECK(z.capacity() == 1, "zero capacity clamped to 1 rather than dividing by zero later");
}

static void test_eviction_releases_payload() {
  std::printf("eviction destroys the payload (the pool-exhaustion risk)\n");
  reset_tracked();
  {
    RingBuffer<Tracked> b(2);
    b.push(10, Tracked(1));
    b.push(20, Tracked(2));
    CHECK(Tracked::live == 2, "two payloads alive while buffered");
    const int before = Tracked::destroyed;
    b.push(30, Tracked(3));                    // evicts ts=10
    CHECK(b.size() == 2, "capacity still respected after eviction");
    CHECK(Tracked::destroyed > before, "EVICTION DESTROYED the rolled-out payload");
    CHECK(Tracked::live == 2, "still exactly two alive -- no accumulation");
    CHECK(b.evicted() == 1, "eviction counted");
  }
  CHECK(Tracked::live == 0, "destructor releases everything on scope exit");
}

static void test_non_monotonic_rejected() {
  std::printf("non-monotonic pushes rejected, not silently misordered\n");
  RingBuffer<int> b(8);
  CHECK(b.push(100, 1), "baseline");
  CHECK(!b.push(100, 2), "equal timestamp rejected");
  CHECK(!b.push(50, 3), "older timestamp rejected");
  CHECK(b.rejected_non_monotonic() == 2, "rejections counted for diagnosis");
  CHECK(b.size() == 1, "rejected items are not stored");
  CHECK(b.push(101, 4), "advancing timestamp still accepted afterwards");

  // The invariant must survive eviction: monotonicity is over the stream's lifetime, not over
  // whatever happens to remain in the buffer.
  RingBuffer<int> c(2);
  c.push(10, 1); c.push(20, 2); c.push(30, 3);      // ts=10 evicted
  CHECK(!c.push(15, 9), "a timestamp older than an EVICTED one is still rejected");
}

static void test_ascending_iteration() {
  std::printf("iteration is ascending\n");
  RingBuffer<int> b(4);
  b.push(5, 1); b.push(7, 2); b.push(9, 3);
  std::vector<int64_t> seen;
  for (auto it = b.begin(); it != b.end(); ++it) seen.push_back(it->timestamp);
  CHECK((seen == std::vector<int64_t>{5, 7, 9}), "ascending order");
}

static void test_take() {
  std::printf("take moves out and frees the slot\n");
  reset_tracked();
  RingBuffer<Tracked> b(4);
  b.push(10, Tracked(1)); b.push(20, Tracked(2)); b.push(30, Tracked(3));
  auto got = b.take(20);
  CHECK(got.has_value(), "found the requested timestamp");
  CHECK(got->value.id == 2, "moved out the right payload");
  CHECK(b.size() == 2, "slot freed");
  CHECK(!b.take(20).has_value(), "second take of the same timestamp finds nothing");
  CHECK(!b.take(999).has_value(), "absent timestamp yields nullopt");
  CHECK(Tracked::live == 3, "taken payload is still alive in the caller's hands, not destroyed");
}

static void test_discard_older_than() {
  std::printf("discard_older_than is STRICT and frees payloads\n");
  reset_tracked();
  RingBuffer<Tracked> b(8);
  for (int i = 1; i <= 5; ++i) b.push(i * 10, Tracked(i));
  const std::size_t freed = b.discard_older_than(30);
  CHECK(freed == 2, "dropped exactly ts=10 and ts=20");
  CHECK(b.size() == 3, "30, 40, 50 retained");
  CHECK(b.begin()->timestamp == 30, "boundary timestamp is KEPT (strictly older only)");
  CHECK(Tracked::live == 3, "discarded payloads were destroyed");
  CHECK(b.discard_older_than(0) == 0, "no-op discard is harmless");
}

// ---------------------------------------------------------------------------------------------
static std::vector<StreamTimestamps> make(std::vector<int64_t> masks,
                                          std::vector<int64_t> depth,
                                          std::vector<int64_t> color,
                                          bool color_required = false) {
  return {{"masks", true,  std::move(masks)},
          {"depth", true,  std::move(depth)},
          {"color", color_required, std::move(color)}};
}

static void test_match_exact_oldest_wins() {
  std::printf("exact match: the OLDEST complete group is chosen\n");
  ExactMatch p;
  MatchResult r;
  // masks lag: depth has run ahead. Two complete groups exist (100 and 200); 100 must win.
  auto streams = make({100, 200}, {100, 200, 300, 400}, {100, 200, 300});
  CHECK(find_oldest_group(streams, p, "masks", &r), "a group is found");
  CHECK(r.group_timestamp == 100, "oldest complete group published");
  CHECK(r.chosen.at("masks") == 100 && r.chosen.at("depth") == 100, "required streams chosen");
  CHECK(r.chosen.count("color") == 1, "optional stream joined when it matched");
}

static void test_match_incomplete() {
  std::printf("no group when a required stream cannot match\n");
  ExactMatch p;
  MatchResult r;
  auto streams = make({150}, {100, 200}, {150});
  CHECK(!find_oldest_group(streams, p, "masks", &r), "no group -- depth has no 150");
  auto empty_ref = make({}, {100}, {100});
  CHECK(!find_oldest_group(empty_ref, p, "masks", &r), "no group when the reference is empty");
}

static void test_optional_does_not_block() {
  std::printf("optional streams never block a group\n");
  ExactMatch p;
  MatchResult r;
  auto streams = make({100}, {100}, {});          // colour absent entirely
  CHECK(find_oldest_group(streams, p, "masks", &r), "group still forms without the optional stream");
  CHECK(r.chosen.count("color") == 0, "absent optional stream is simply not in the group");
  CHECK(r.group_timestamp == 100, "group timestamp unaffected by the missing optional stream");

  // ... but a required colour DOES block.
  auto req = make({100}, {100}, {}, /*color_required=*/true);
  CHECK(!find_oldest_group(req, p, "masks", &r), "required colour blocks when absent");
}

static void test_window_match_and_tiebreak() {
  std::printf("window match: nearest wins, older breaks a tie, deterministically\n");
  WindowMatch p(10);
  MatchResult r;
  // depth has 95 and 104, both within +/-10 of 100. 104 is nearer: |100-104|=4 vs |100-95|=5.
  // (Note 96 and 104 would be EQUIDISTANT at 4 apiece -- an easy arithmetic slip, and the reason
  // the tie-break below is tested separately.)
  auto streams = make({100}, {95, 104}, {});
  CHECK(find_oldest_group(streams, p, "masks", &r), "windowed group found");
  CHECK(r.chosen.at("depth") == 104, "nearest candidate chosen");
  CHECK(r.group_timestamp == 100, "group timestamp is the MINIMUM chosen, so no member is discarded");

  // Equidistant: 95 and 105 are both 5 away -> the older must win, regardless of order.
  auto tie  = make({100}, {95, 105}, {});
  auto tie2 = make({100}, {105, 95}, {});   // note: not ascending, but the tie-break must not care
  MatchResult a, b;
  CHECK(find_oldest_group(tie,  p, "masks", &a), "tie group found");
  CHECK(find_oldest_group(tie2, p, "masks", &b), "tie group found with inputs reversed");
  CHECK(a.chosen.at("depth") == 95, "older wins an equidistant tie");
  CHECK(a.chosen.at("depth") == b.chosen.at("depth"), "tie-break is order-independent");

  WindowMatch narrow(1);
  auto outside = make({100}, {96, 104}, {});
  CHECK(!find_oldest_group(outside, narrow, "masks", &r), "nothing matches outside the window");
}

static void test_group_timestamp_is_safe_discard_boundary() {
  std::printf("group timestamp is a safe discard boundary\n");
  WindowMatch p(20);
  MatchResult r;
  auto streams = make({100}, {85}, {90});
  CHECK(find_oldest_group(streams, p, "masks", &r), "group found across a window");
  CHECK(r.group_timestamp == 85, "boundary is the oldest MEMBER, not the reference");
  // Discarding strictly-older-than 85 in a buffer holding 85 must keep it.
  RingBuffer<int> depth(4);
  depth.push(85, 1);
  CHECK(depth.discard_older_than(r.group_timestamp) == 0,
        "no member of the published group is discarded by its own boundary");
}

static void test_end_to_end_lagging_reference() {
  std::printf("end to end: a lagging reference with a fast partner (the real shape)\n");
  // Depth at ~30 fps, masks ~5x slower and 2 frames behind -- the measured situation.
  RingBuffer<Tracked> depth(12), masks(4);
  reset_tracked();
  for (int i = 0; i < 12; ++i) depth.push(1000 + i * 33, Tracked(i));
  masks.push(1000, Tracked(100));            // mask for the OLDEST depth frame
  masks.push(1000 + 6 * 33, Tracked(106));

  std::vector<int64_t> dts, mts;
  for (auto it = depth.begin(); it != depth.end(); ++it) dts.push_back(it->timestamp);
  for (auto it = masks.begin(); it != masks.end(); ++it) mts.push_back(it->timestamp);

  ExactMatch p;
  MatchResult r;
  CHECK(find_oldest_group({{"masks", true, mts}, {"depth", true, dts}}, p, "masks", &r),
        "group found despite the mask stream lagging");
  CHECK(r.group_timestamp == 1000, "oldest pair published first");

  // Publish it, then apply the discard rule.
  auto m = masks.take(r.chosen.at("masks"));
  auto d = depth.take(r.chosen.at("depth"));
  CHECK(m.has_value() && d.has_value(), "both members taken out");
  masks.discard_older_than(r.group_timestamp);
  depth.discard_older_than(r.group_timestamp);
  CHECK(masks.size() == 1, "the FUTURE mask survives the discard");
  CHECK(depth.size() == 11, "depth keeps everything newer than the published group");

  // The second group must now be discoverable -- i.e. the first publish did not starve it.
  std::vector<int64_t> dts2, mts2;
  for (auto it = depth.begin(); it != depth.end(); ++it) dts2.push_back(it->timestamp);
  for (auto it = masks.begin(); it != masks.end(); ++it) mts2.push_back(it->timestamp);
  MatchResult r2;
  CHECK(find_oldest_group({{"masks", true, mts2}, {"depth", true, dts2}}, p, "masks", &r2),
        "the next group is still matchable after discarding");
  CHECK(r2.group_timestamp == 1000 + 6 * 33, "and it is the next-oldest one");
}

int main() {
  std::printf("== temporal-sync logic tests ==\n\n");
  test_push_size_capacity();
  test_eviction_releases_payload();
  test_non_monotonic_rejected();
  test_ascending_iteration();
  test_take();
  test_discard_older_than();
  test_match_exact_oldest_wins();
  test_match_incomplete();
  test_optional_does_not_block();
  test_window_match_and_tiebreak();
  test_group_timestamp_is_safe_discard_boundary();
  test_end_to_end_lagging_reference();
  std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
