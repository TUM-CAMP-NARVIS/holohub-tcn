// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>

#include "iox2/iceoryx2.hpp"

#include "generated/shm_parameter_rpc.capnp.h"

namespace tcn::shm {

// Type aliases for iceoryx2 request-response with byte slices
static constexpr iox2::ServiceType RpcServiceType = iox2::ServiceType::Ipc;
using RpcSlice = iox2::bb::Slice<uint8_t>;

/// Represents a parameter schema entry for a single node parameter.
struct ParameterSchemaEntry {
    std::string key;
    tcnart_msgs::rpc::ParameterValueType valueType;
    // validEntries left empty for now (matching Python impl)
};

/// Represents a parameter value for a single node parameter.
struct ParameterEntry {
    std::string key;
    // The value is stored as a variant-like structure.
    // For simplicity, we use a capnp message builder.
    tcnart_msgs::rpc::ParameterValueType type;
    // Union of possible values
    int32_t int32Value{0};
    int64_t int64Value{0};
    float floatValue{0.0f};
    double doubleValue{0.0};
    bool boolValue{false};
    std::string stringValue;
};

/// Parameter RPC Server - listens for Cap'n Proto RPC requests over iceoryx2 SHM.
///
/// Port of Python ParameterRpcServer. Creates an iceoryx2 request-response service
/// and handles ListEntities, GetParameterSchema, GetParameterValues, SetParameterValues.
class ParameterRpcServer {
public:
    using NodeSchemaMap = std::map<std::string, std::vector<ParameterSchemaEntry>>;
    using GetValuesCallback = std::function<std::vector<ParameterEntry>(const std::string& entity_name)>;
    using SetValuesCallback = std::function<bool(const std::string& entity_name,
                                                  const std::vector<ParameterEntry>& values)>;

    ParameterRpcServer(iox2::Node<RpcServiceType>& node,
                       const std::string& rpc_service_name);

    ~ParameterRpcServer() = default;
    ParameterRpcServer(const ParameterRpcServer&) = delete;
    ParameterRpcServer& operator=(const ParameterRpcServer&) = delete;

    /// Register parameter schema for a named entity (node).
    void register_entity(const std::string& entity_name,
                         std::vector<ParameterSchemaEntry> schema);

    /// Set callback for getting parameter values.
    void set_get_values_callback(GetValuesCallback cb) { get_values_cb_ = std::move(cb); }

    /// Set callback for setting parameter values.
    void set_set_values_callback(SetValuesCallback cb) { set_values_cb_ = std::move(cb); }

    /// List registered entity names (sorted).
    std::vector<std::string> list_nodes() const;

    /// Get parameter schema for an entity.
    const std::vector<ParameterSchemaEntry>& get_parameter_schema(const std::string& entity_name) const;

    /// Signal the server to stop.
    void shutdown() { should_stop_.store(true); }

    /// Blocking serve loop - processes requests until shutdown() is called.
    void serve_blocking();

private:
    /// Handle a single decoded RPC request, returns serialized response bytes.
    kj::Array<capnp::word> handle_request(tcnart_msgs::rpc::ParameterRpcRequest::Reader request);

    /// Build a ListEntities response.
    kj::Array<capnp::word> build_list_entities_response(
        tcnart_msgs::rpc::RPCCommandType command_type);

    /// Build a GetParameterSchema response.
    kj::Array<capnp::word> build_get_schema_response(
        tcnart_msgs::rpc::RPCCommandType command_type,
        const std::string& entity_name);

    /// Build a GetParameterValues response.
    kj::Array<capnp::word> build_get_values_response(
        tcnart_msgs::rpc::RPCCommandType command_type,
        const std::string& entity_name);

    /// Build an error response.
    kj::Array<capnp::word> build_error_response(
        tcnart_msgs::rpc::RPCCommandType command_type);

    std::atomic<bool> should_stop_{false};
    NodeSchemaMap node_schema_;
    GetValuesCallback get_values_cb_;
    SetValuesCallback set_values_cb_;

    // iceoryx2 service and server - stored as optional to allow deferred init
    using ServerType = iox2::Server<RpcServiceType, RpcSlice, void, RpcSlice, void>;
    std::unique_ptr<ServerType> server_;

    static const std::vector<ParameterSchemaEntry> empty_schema_;
};

/// Parameter RPC Client - sends Cap'n Proto RPC requests over iceoryx2 SHM.
///
/// Port of Python ParameterRpcClient.
class ParameterRpcClient {
public:
    ParameterRpcClient(iox2::Node<RpcServiceType>& node,
                       const std::string& rpc_service_name);

    ~ParameterRpcClient() = default;
    ParameterRpcClient(const ParameterRpcClient&) = delete;
    ParameterRpcClient& operator=(const ParameterRpcClient&) = delete;

    /// List available entities/components.
    std::vector<std::string> list_components(int timeout_ms = 1000);

    /// Get parameter schema for an entity.
    std::vector<ParameterSchemaEntry> get_parameter_schema(const std::string& entity_name,
                                                            int timeout_ms = 1000);

    /// Get current parameter values for an entity.
    std::vector<ParameterEntry> get_parameter_values(const std::string& entity_name,
                                                      int timeout_ms = 1000);

    /// Set parameter values for an entity.
    bool set_parameter_values(const std::string& entity_name,
                              const std::vector<ParameterEntry>& values,
                              int timeout_ms = 1000);

private:
    /// Send a serialized request and receive a response, with timeout.
    /// Returns the deserialized response, or nullopt on timeout.
    template <typename ResponseHandler>
    auto call_remote(capnp::MessageBuilder& request_builder,
                     ResponseHandler handler,
                     int timeout_ms) -> decltype(handler(std::declval<tcnart_msgs::rpc::ParameterRpcResponse::Reader>()));

    using ClientType = iox2::Client<RpcServiceType, RpcSlice, void, RpcSlice, void>;
    std::unique_ptr<ClientType> client_;
};

}  // namespace tcn::shm
