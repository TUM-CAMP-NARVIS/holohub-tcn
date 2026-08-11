/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TCN_STREAM_SYNCHRONIZER_RING_BUFFER_HPP
#define TCN_STREAM_SYNCHRONIZER_RING_BUFFER_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>

namespace tcn::sync {

/// One buffered item: an acquisition timestamp and the payload it belongs to.
template <typename T>
struct Timed {
  int64_t timestamp = 0;
  T value{};
};

/**
 * @brief Fixed-capacity, timestamp-ordered store of movable payloads, one per stream.
 *
 * TEMPLATED ON THE PAYLOAD for a specific reason: the operator instantiates it with
 * `holoscan::gxf::Entity`, but the interesting behaviour here is lifetime and ordering, neither of
 * which needs Holoscan. Keeping the payload abstract lets the whole class be tested on a host with
 * no GPU and no SDK, against a mock whose destructor is observable -- which is how the dominant
 * risk (see below) is actually checked rather than argued about.
 *
 * DOMINANT RISK: these payloads own pooled device memory. If eviction overwrites a slot without
 * destroying the previous payload, the underlying GXF entity stays alive and its tensors are never
 * returned to the `BlockMemoryPool`. The failure mode is not growing RSS -- it is the pool running
 * dry and the graph stalling, which looks like a hang rather than a leak.
 *
 * That is why this is backed by a `std::deque` with an enforced capacity rather than a hand-rolled
 * circular array over raw storage. Eviction is `pop_front()`, so the payload's destructor runs by
 * construction and there is no slot to forget to reset. A circular array would be marginally
 * cheaper and materially easier to get wrong; at these sizes (single to low double digits) the
 * difference is irrelevant next to one H2D copy.
 *
 * ORDERING is an invariant, not an assumption. `push` rejects any timestamp that does not advance,
 * counting it, rather than inserting out of order -- because the matcher relies on ascending
 * iteration to return the OLDEST complete group first, and a silently misordered element would
 * break that guarantee in a way that is very hard to see downstream.
 */
template <typename T>
class RingBuffer {
 public:
  using container = std::deque<Timed<T>>;
  using const_iterator = typename container::const_iterator;

  explicit RingBuffer(std::size_t capacity)
      : capacity_(capacity == 0 ? 1 : capacity) {}

  /**
   * @brief Insert a payload, evicting the oldest if full.
   *
   * @return true if stored; false if rejected because `timestamp` did not advance.
   */
  bool push(int64_t timestamp, T&& value) {
    if (have_pushed_ && timestamp <= last_pushed_) {
      ++rejected_non_monotonic_;
      return false;
    }
    while (items_.size() >= capacity_) {
      items_.pop_front();          // destroys the payload -> handle released
      ++evicted_;
    }
    items_.push_back(Timed<T>{timestamp, std::move(value)});
    last_pushed_ = timestamp;
    have_pushed_ = true;
    return true;
  }

  /// Move out the item with exactly `timestamp`, freeing its slot. nullopt if absent.
  std::optional<Timed<T>> take(int64_t timestamp) {
    for (auto it = items_.begin(); it != items_.end(); ++it) {
      if (it->timestamp == timestamp) {
        Timed<T> out = std::move(*it);
        items_.erase(it);
        return out;
      }
      if (it->timestamp > timestamp) break;   // ascending: cannot appear later
    }
    return std::nullopt;
  }

  /// Drop everything STRICTLY older than `timestamp`. Returns how many were freed.
  std::size_t discard_older_than(int64_t timestamp) {
    std::size_t n = 0;
    while (!items_.empty() && items_.front().timestamp < timestamp) {
      items_.pop_front();
      ++n;
    }
    return n;
  }

  /// Ascending-timestamp iteration, for the matcher.
  const_iterator begin() const { return items_.begin(); }
  const_iterator end()   const { return items_.end(); }

  std::size_t size()     const { return items_.size(); }
  std::size_t capacity() const { return capacity_; }
  bool        empty()    const { return items_.empty(); }
  bool        full()     const { return items_.size() >= capacity_; }

  /// Diagnostics. Non-zero `rejected_non_monotonic` means a stream is not behaving as assumed;
  /// a climbing `evicted` means this stream is outrunning the group it must be matched against.
  std::size_t rejected_non_monotonic() const { return rejected_non_monotonic_; }
  std::size_t evicted()                const { return evicted_; }

  void clear() { items_.clear(); }

 private:
  container   items_;
  std::size_t capacity_;
  int64_t     last_pushed_ = 0;
  bool        have_pushed_ = false;
  std::size_t rejected_non_monotonic_ = 0;
  std::size_t evicted_ = 0;
};

}  // namespace tcn::sync

#endif  // TCN_STREAM_SYNCHRONIZER_RING_BUFFER_HPP
