// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace tcn::shm {

enum class PubSubEvent : uint8_t {
    PublisherConnected = 0,
    PublisherDisconnected = 1,
    SubscriberConnected = 2,
    SubscriberDisconnected = 3,
    SentSample = 4,
    ReceivedSample = 5,
    SentHistory = 6,
    Unknown = 7,
};

struct ShmSerializedMessage {
    static constexpr const char* IOX2_TYPE_NAME = "ShmSerializedMessage";
    static constexpr std::size_t payloadMaxSize = 16384;

    std::array<unsigned char, payloadMaxSize> payload{};
    std::size_t payloadSize{0};

    /// Return a view of the valid payload bytes.
    std::string_view payload_view() const noexcept {
        auto n = payloadSize;
        if (n > payloadMaxSize) n = payloadMaxSize;
        return {reinterpret_cast<const char*>(payload.data()), n};
    }

    const unsigned char* payload_data() const noexcept { return payload.data(); }
    std::size_t payload_len() const noexcept {
        return payloadSize > payloadMaxSize ? payloadMaxSize : payloadSize;
    }
};

struct ShmSerializedStreamHeader {
    static constexpr const char* IOX2_TYPE_NAME = "ShmSerializedStreamHeader";
    static constexpr std::size_t payloadMaxSize = 1024;

    std::array<unsigned char, payloadMaxSize> payload{};
    std::size_t payloadSize{0};
    uint64_t timestamp{0};

    std::string_view payload_view() const noexcept {
        auto n = payloadSize;
        if (n > payloadMaxSize) n = payloadMaxSize;
        return {reinterpret_cast<const char*>(payload.data()), n};
    }

    const unsigned char* payload_data() const noexcept { return payload.data(); }
    std::size_t payload_len() const noexcept {
        return payloadSize > payloadMaxSize ? payloadMaxSize : payloadSize;
    }
};

}  // namespace tcn::shm
