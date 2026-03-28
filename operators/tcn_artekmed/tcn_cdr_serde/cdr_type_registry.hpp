// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "cdr_serde.hpp"

namespace tcn::cdr {

/// Decoded CDR result: raw field bytes + metadata key-value pairs.
struct DecodedMessage {
    std::string type_name;
    std::vector<uint8_t> payload;  // Primary data (e.g. image bytes)
    std::unordered_map<std::string, std::string> metadata;
};

/// Type-erased deserializer function: takes raw CDR bytes, returns DecodedMessage.
using DeserializeFn = std::function<bool(const uint8_t*, size_t, DecodedMessage&)>;

/// Registry that maps CDR type name strings to deserializer functions.
/// Mirrors the Python _TYPE_REGISTRY pattern from tcnart's cdr_serialization.py.
class CdrTypeRegistry {
 public:
    static CdrTypeRegistry& instance() {
        static CdrTypeRegistry reg;
        return reg;
    }

    void register_type(const std::string& type_name, DeserializeFn fn) {
        registry_[type_name] = std::move(fn);
    }

    bool has_type(const std::string& type_name) const {
        return registry_.count(type_name) > 0;
    }

    bool decode(const std::string& type_name,
                const uint8_t* data, size_t size,
                DecodedMessage& out) const {
        auto it = registry_.find(type_name);
        if (it == registry_.end()) return false;
        out.type_name = type_name;
        return it->second(data, size, out);
    }

    bool decode(const std::string& type_name,
                const std::vector<uint8_t>& data,
                DecodedMessage& out) const {
        return decode(type_name, data.data(), data.size(), out);
    }

    const std::unordered_map<std::string, DeserializeFn>& types() const {
        return registry_;
    }

 private:
    CdrTypeRegistry() = default;
    std::unordered_map<std::string, DeserializeFn> registry_;
};

/// RAII helper for registering a type at static-init time.
struct CdrTypeRegistrar {
    CdrTypeRegistrar(const std::string& name, DeserializeFn fn) {
        CdrTypeRegistry::instance().register_type(name, std::move(fn));
    }
};

}  // namespace tcn::cdr
