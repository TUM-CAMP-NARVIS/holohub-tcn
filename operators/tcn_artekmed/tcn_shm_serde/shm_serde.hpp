// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <capnp/serialize-packed.h>

#include "generated/shm_synchronized_transport.capnp.h"
#include "generated/shm_parameter_rpc.capnp.h"

namespace tcn::shm {

/// Decode a Cap'n Proto message from a raw byte buffer.
/// Tries standard (unpacked) encoding first, falls back to packed encoding.
template <typename T>
std::optional<typename T::Reader> decode_message(
    capnp::FlatArrayMessageReader*& reader_out,
    const unsigned char* data,
    std::size_t size) {
    // This template approach won't work well because readers own their data.
    // Use the non-template overloads below instead.
    return std::nullopt;
}

/// Result of a decode operation: owns the message reader and provides typed access.
template <typename T>
class DecodedMessage {
public:
    DecodedMessage() = default;

    /// Decode from raw bytes (unpacked encoding first, packed fallback).
    static DecodedMessage decode(const unsigned char* data, std::size_t size) {
        DecodedMessage result;

        // Copy the raw data so the FlatArrayMessageReader references memory we own,
        // not the caller's buffer (which may be released after this call returns).
        auto aligned_size = (size + sizeof(capnp::word) - 1) / sizeof(capnp::word);
        result.owned_data_.resize(aligned_size * sizeof(capnp::word), 0);
        std::memcpy(result.owned_data_.data(), data, size);

        auto words = kj::arrayPtr(
            reinterpret_cast<const capnp::word*>(result.owned_data_.data()),
            aligned_size);
        try {
            result.reader_ = std::make_unique<capnp::FlatArrayMessageReader>(words);
            // Validate by accessing root - will throw on corruption
            (void)result.reader_->template getRoot<T>();
            return result;
        } catch (...) {
            // Fall back to packed encoding
            result.reader_.reset();
        }

        try {
            auto bytes = kj::arrayPtr(result.owned_data_.data(), size);
            kj::ArrayInputStream stream(bytes);
            result.packed_reader_ = std::make_unique<capnp::PackedMessageReader>(stream);
            (void)result.packed_reader_->template getRoot<T>();
            return result;
        } catch (...) {
            result.packed_reader_.reset();
            throw std::runtime_error("Failed to decode Cap'n Proto message (tried both encodings)");
        }
    }

    /// Access the decoded message root.
    typename T::Reader root() const {
        if (reader_) return reader_->template getRoot<T>();
        if (packed_reader_) return packed_reader_->template getRoot<T>();
        throw std::runtime_error("No decoded message available");
    }

    explicit operator bool() const { return reader_ != nullptr || packed_reader_ != nullptr; }

private:
    std::vector<unsigned char> owned_data_;  // Owned copy of raw bytes backing the reader
    mutable std::unique_ptr<capnp::FlatArrayMessageReader> reader_;
    mutable std::unique_ptr<capnp::PackedMessageReader> packed_reader_;
};

// Convenience type aliases for the main message types
using DecodedBufferDescriptor = DecodedMessage<artekmed::shm::ShmBufferDescriptor>;
using DecodedConnectionStatus = DecodedMessage<artekmed::shm::ShmBufferConnectionStatus>;
using DecodedDeviceContext = DecodedMessage<artekmed::shm::ShmDeviceContext>;
using DecodedRpcRequest = DecodedMessage<tcnart_msgs::rpc::ParameterRpcRequest>;
using DecodedRpcResponse = DecodedMessage<tcnart_msgs::rpc::ParameterRpcResponse>;

/// Decode an ShmBufferDescriptor from raw bytes.
inline DecodedBufferDescriptor decode_buffer_descriptor(const unsigned char* data, std::size_t size) {
    return DecodedBufferDescriptor::decode(data, size);
}

/// Decode an ShmBufferConnectionStatus from raw bytes.
inline DecodedConnectionStatus decode_shm_buffer_connection_status(const unsigned char* data, std::size_t size) {
    return DecodedConnectionStatus::decode(data, size);
}

/// Decode an ShmDeviceContext from raw bytes.
inline DecodedDeviceContext decode_shm_device_context(const unsigned char* data, std::size_t size) {
    return DecodedDeviceContext::decode(data, size);
}

/// Decode a ParameterRpcRequest from raw bytes.
inline DecodedRpcRequest decode_rpc_request(const unsigned char* data, std::size_t size) {
    return DecodedRpcRequest::decode(data, size);
}

/// Decode a ParameterRpcResponse from raw bytes.
inline DecodedRpcResponse decode_rpc_response(const unsigned char* data, std::size_t size) {
    return DecodedRpcResponse::decode(data, size);
}

/// Serialize a Cap'n Proto message builder to bytes.
inline kj::Array<capnp::word> serialize_message(capnp::MessageBuilder& builder) {
    return capnp::messageToFlatArray(builder);
}

}  // namespace tcn::shm
