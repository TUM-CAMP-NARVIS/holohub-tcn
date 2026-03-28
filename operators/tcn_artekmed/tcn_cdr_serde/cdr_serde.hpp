// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <fastcdr/Cdr.h>
#include <fastcdr/FastBuffer.h>

namespace tcn::cdr {

/// Read a CDR-encoded message from raw bytes.
/// Uses DDS-CDR encoding with encapsulation header (4 bytes).
class CdrBufferReader {
public:
    /// Deserialize a CDR payload into the given message struct.
    /// @tparam MT  FastCDR-compatible message type (operator>> overload).
    /// @param data Raw CDR bytes (including 4-byte encapsulation header).
    /// @param size Number of bytes.
    /// @param value Output message struct (must be default-constructed).
    /// @return true on success, false on deserialization error.
    template <typename MT>
    bool read(const uint8_t* data, size_t size, MT& value) {
        try {
            eprosima::fastcdr::FastBuffer buffer(
                const_cast<char*>(reinterpret_cast<const char*>(data)),
                size);
            eprosima::fastcdr::Cdr cdr_des(
                buffer,
                eprosima::fastcdr::Cdr::DEFAULT_ENDIAN,
                eprosima::fastcdr::CdrVersion::DDS_CDR);
            cdr_des.read_encapsulation();
            cdr_des >> value;
            return true;
        } catch (...) {
            return false;
        }
    }

    /// Convenience overload that takes a vector.
    template <typename MT>
    bool read(const std::vector<uint8_t>& data, MT& value) {
        return read(data.data(), data.size(), value);
    }
};

/// Write a CDR-encoded message to bytes.
/// Uses DDS-CDR encoding with encapsulation header.
class CdrBufferWriter {
public:
    /// Serialize a message struct to CDR bytes.
    /// @tparam MT  FastCDR-compatible message type (operator<< overload).
    /// @param value The message to serialize.
    /// @return CDR-encoded bytes (including encapsulation header).
    template <typename MT>
    std::vector<uint8_t> write(const MT& value) {
        // Pre-allocate buffer (resize if needed by FastBuffer)
        buffer_.resize(4096);
        eprosima::fastcdr::FastBuffer fast_buffer(
            reinterpret_cast<char*>(buffer_.data()),
            buffer_.size());
        eprosima::fastcdr::Cdr cdr_ser(
            fast_buffer,
            eprosima::fastcdr::Cdr::DEFAULT_ENDIAN,
            eprosima::fastcdr::CdrVersion::DDS_CDR);
        cdr_ser.serialize_encapsulation();
        cdr_ser << value;

        auto serialized_length = cdr_ser.get_serialized_data_length();
        return std::vector<uint8_t>(buffer_.begin(),
                                     buffer_.begin() + serialized_length);
    }

private:
    std::vector<uint8_t> buffer_;
};

}  // namespace tcn::cdr
