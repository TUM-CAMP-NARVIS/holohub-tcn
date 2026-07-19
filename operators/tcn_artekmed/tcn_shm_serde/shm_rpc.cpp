// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "shm_rpc.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include <capnp/serialize.h>

#include "iox2/iceoryx2.hpp"

namespace tcn::shm {

const std::vector<ParameterSchemaEntry> ParameterRpcServer::empty_schema_;

// ---------------------------------------------------------------------------
// Helpers: write/read capnp message to/from iceoryx2 slice payload
// ---------------------------------------------------------------------------

namespace {

/// Write serialized capnp words into an iceoryx2 slice payload.
void write_capnp_to_slice(uint8_t* dst, std::size_t dst_len,
                           const kj::Array<capnp::word>& words) {
    auto bytes = words.asBytes();
    auto copy_len = std::min(dst_len, bytes.size());
    std::memcpy(dst, bytes.begin(), copy_len);
}

/// Build a ParameterValue capnp field from a ParameterEntry.
void set_parameter_value(tcnart_msgs::rpc::ParameterValue::Builder builder,
                         const ParameterEntry& entry) {
    switch (entry.type) {
    case tcnart_msgs::rpc::ParameterValueType::PVT_INT32:
        builder.setInt32Value(entry.int32Value);
        break;
    case tcnart_msgs::rpc::ParameterValueType::PVT_INT64:
        builder.setInt64Value(entry.int64Value);
        break;
    case tcnart_msgs::rpc::ParameterValueType::PVT_FLOAT:
        builder.setFloatValue(entry.floatValue);
        break;
    case tcnart_msgs::rpc::ParameterValueType::PVT_DOUBLE:
        builder.setDoubleValue(entry.doubleValue);
        break;
    case tcnart_msgs::rpc::ParameterValueType::PVT_BOOL:
        builder.setBoolValue(entry.boolValue);
        break;
    case tcnart_msgs::rpc::ParameterValueType::PVT_STRING:
        builder.setStringValue(entry.stringValue);
        break;
    default:
        break;
    }
}

/// Read a ParameterEntry from a capnp Parameter reader.
ParameterEntry read_parameter(tcnart_msgs::rpc::Parameter::Reader param) {
    ParameterEntry entry;
    entry.key = param.getKey();
    auto val = param.getValue();
    switch (val.which()) {
    case tcnart_msgs::rpc::ParameterValue::INT32_VALUE:
        entry.type = tcnart_msgs::rpc::ParameterValueType::PVT_INT32;
        entry.int32Value = val.getInt32Value();
        break;
    case tcnart_msgs::rpc::ParameterValue::INT64_VALUE:
        entry.type = tcnart_msgs::rpc::ParameterValueType::PVT_INT64;
        entry.int64Value = val.getInt64Value();
        break;
    case tcnart_msgs::rpc::ParameterValue::FLOAT_VALUE:
        entry.type = tcnart_msgs::rpc::ParameterValueType::PVT_FLOAT;
        entry.floatValue = val.getFloatValue();
        break;
    case tcnart_msgs::rpc::ParameterValue::DOUBLE_VALUE:
        entry.type = tcnart_msgs::rpc::ParameterValueType::PVT_DOUBLE;
        entry.doubleValue = val.getDoubleValue();
        break;
    case tcnart_msgs::rpc::ParameterValue::BOOL_VALUE:
        entry.type = tcnart_msgs::rpc::ParameterValueType::PVT_BOOL;
        entry.boolValue = val.getBoolValue();
        break;
    case tcnart_msgs::rpc::ParameterValue::STRING_VALUE:
        entry.type = tcnart_msgs::rpc::ParameterValueType::PVT_STRING;
        entry.stringValue = val.getStringValue();
        break;
    default:
        entry.type = tcnart_msgs::rpc::ParameterValueType::PVT_STRING;
        break;
    }
    return entry;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// ParameterRpcServer
// ---------------------------------------------------------------------------

ParameterRpcServer::ParameterRpcServer(iox2::Node<RpcServiceType>& node,
                                       const std::string& rpc_service_name) {
    auto sname = iox2::ServiceName::create(rpc_service_name.c_str()).value();
    auto service = node.service_builder(sname)
                       .request_response<RpcSlice, RpcSlice>()
                       .open_or_create()
                       .value();
    auto server = service.server_builder()
                      .initial_max_slice_len(16)
                      .allocation_strategy(iox2::AllocationStrategy::PowerOfTwo)
                      .create()
                      .value();
    server_ = std::make_unique<ServerType>(std::move(server));
}

void ParameterRpcServer::register_entity(const std::string& entity_name,
                                          std::vector<ParameterSchemaEntry> schema) {
    node_schema_[entity_name] = std::move(schema);
}

std::vector<std::string> ParameterRpcServer::list_nodes() const {
    std::vector<std::string> names;
    names.reserve(node_schema_.size());
    for (const auto& [name, _] : node_schema_) {
        names.push_back(name);
    }
    // std::map is already sorted by key
    return names;
}

const std::vector<ParameterSchemaEntry>& ParameterRpcServer::get_parameter_schema(
    const std::string& entity_name) const {
    auto it = node_schema_.find(entity_name);
    if (it == node_schema_.end()) return empty_schema_;
    return it->second;
}

void ParameterRpcServer::serve_blocking() {
    auto cycle_time = iox2::bb::Duration::from_millis(5);
    while (!should_stop_.load()) {
        // Process all pending requests
        while (!should_stop_.load()) {
            auto maybe_request = server_->receive();
            if (!maybe_request.has_value()) break;
            auto& opt_request = maybe_request.value();
            if (!opt_request.has_value()) break;
            auto& active_request = opt_request.value();

            // Read request payload
            auto request_slice = active_request.template payload<RpcSlice>();
            auto request_data = request_slice.data();
            auto request_len = request_slice.number_of_elements();

            // Decode the request
            kj::Array<capnp::word> response_words;
            try {
                auto words = kj::arrayPtr(
                    reinterpret_cast<const capnp::word*>(request_data),
                    request_len / sizeof(capnp::word));
                capnp::FlatArrayMessageReader reader(words);
                auto request = reader.getRoot<tcnart_msgs::rpc::ParameterRpcRequest>();
                response_words = handle_request(request);
            } catch (...) {
                // Build error response
                capnp::MallocMessageBuilder builder;
                auto resp = builder.initRoot<tcnart_msgs::rpc::ParameterRpcResponse>();
                resp.setResponseType(tcnart_msgs::rpc::RPCResponseStatus::RPC_STATUS_ERROR);
                response_words = capnp::messageToFlatArray(builder);
            }

            // Send response
            auto response_bytes = response_words.asBytes();
            auto loan = active_request.loan_slice_uninit(response_bytes.size());
            if (loan.has_value()) {
                auto& response_uninit = loan.value();
                auto dst = response_uninit.payload_mut().data();
                std::memcpy(dst, response_bytes.begin(), response_bytes.size());
                auto response_init = iox2::assume_init(std::move(response_uninit));
                iox2::send(std::move(response_init)).value();
            }
        }

        // Brief sleep to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

kj::Array<capnp::word> ParameterRpcServer::handle_request(
    tcnart_msgs::rpc::ParameterRpcRequest::Reader request) {
    auto cmd = request.getCommandType();
    switch (cmd) {
    case tcnart_msgs::rpc::RPCCommandType::RPC_COMMAND_LIST_ENTITIES:
        return build_list_entities_response(cmd);

    case tcnart_msgs::rpc::RPCCommandType::RPC_COMMAND_GET_PARAMETER_SCHEMA:
        return build_get_schema_response(cmd, request.getEntityName());

    case tcnart_msgs::rpc::RPCCommandType::RPC_COMMAND_GET_PARAMETER_VALUES:
        return build_get_values_response(cmd, request.getEntityName());

    case tcnart_msgs::rpc::RPCCommandType::RPC_COMMAND_SET_PARAMETER_VALUES: {
        // SetParameterValues - delegate to callback if available
        if (set_values_cb_) {
            std::vector<ParameterEntry> values;
            if (request.hasPayload()) {
                for (auto param : request.getPayload().getParameters()) {
                    values.push_back(read_parameter(param));
                }
            }
            set_values_cb_(request.getEntityName(), values);
        }
        // Return success response (matching Python stub behavior)
        capnp::MallocMessageBuilder builder;
        auto resp = builder.initRoot<tcnart_msgs::rpc::ParameterRpcResponse>();
        resp.setCommandType(cmd);
        resp.setResponseType(tcnart_msgs::rpc::RPCResponseStatus::RPC_STATUS_SUCCESS);
        return capnp::messageToFlatArray(builder);
    }

    default:
        return build_error_response(cmd);
    }
}

kj::Array<capnp::word> ParameterRpcServer::build_list_entities_response(
    tcnart_msgs::rpc::RPCCommandType command_type) {
    auto names = list_nodes();
    capnp::MallocMessageBuilder builder;
    auto resp = builder.initRoot<tcnart_msgs::rpc::ParameterRpcResponse>();
    resp.setCommandType(command_type);
    resp.setResponseType(tcnart_msgs::rpc::RPCResponseStatus::RPC_STATUS_SUCCESS);
    auto list = resp.initEntitiesList(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        list.set(i, names[i]);
    }
    return capnp::messageToFlatArray(builder);
}

kj::Array<capnp::word> ParameterRpcServer::build_get_schema_response(
    tcnart_msgs::rpc::RPCCommandType command_type,
    const std::string& entity_name) {
    const auto& schema = get_parameter_schema(entity_name);
    capnp::MallocMessageBuilder builder;
    auto resp = builder.initRoot<tcnart_msgs::rpc::ParameterRpcResponse>();
    resp.setCommandType(command_type);
    resp.setResponseType(tcnart_msgs::rpc::RPCResponseStatus::RPC_STATUS_SUCCESS);
    auto ps = resp.initParameterSchema();
    auto params = ps.initParameters(schema.size());
    for (size_t i = 0; i < schema.size(); ++i) {
        params[i].setKey(schema[i].key);
        params[i].setValueType(schema[i].valueType);
        params[i].initValidEntries(0);
    }
    return capnp::messageToFlatArray(builder);
}

kj::Array<capnp::word> ParameterRpcServer::build_get_values_response(
    tcnart_msgs::rpc::RPCCommandType command_type,
    const std::string& entity_name) {
    capnp::MallocMessageBuilder builder;
    auto resp = builder.initRoot<tcnart_msgs::rpc::ParameterRpcResponse>();
    resp.setCommandType(command_type);
    resp.setResponseType(tcnart_msgs::rpc::RPCResponseStatus::RPC_STATUS_SUCCESS);

    std::vector<ParameterEntry> values;
    if (get_values_cb_) {
        values = get_values_cb_(entity_name);
    }
    auto pv = resp.initParameterValues();
    auto params = pv.initParameters(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        params[i].setKey(values[i].key);
        set_parameter_value(params[i].initValue(), values[i]);
    }
    return capnp::messageToFlatArray(builder);
}

kj::Array<capnp::word> ParameterRpcServer::build_error_response(
    tcnart_msgs::rpc::RPCCommandType command_type) {
    capnp::MallocMessageBuilder builder;
    auto resp = builder.initRoot<tcnart_msgs::rpc::ParameterRpcResponse>();
    resp.setCommandType(command_type);
    resp.setResponseType(tcnart_msgs::rpc::RPCResponseStatus::RPC_STATUS_ERROR);
    return capnp::messageToFlatArray(builder);
}

// ---------------------------------------------------------------------------
// ParameterRpcClient
// ---------------------------------------------------------------------------

ParameterRpcClient::ParameterRpcClient(iox2::Node<RpcServiceType>& node,
                                       const std::string& rpc_service_name) {
    auto sname = iox2::ServiceName::create(rpc_service_name.c_str()).value();
    auto service = node.service_builder(sname)
                       .request_response<RpcSlice, RpcSlice>()
                       .open_or_create()
                       .value();
    auto client = service.client_builder()
                      .initial_max_slice_len(16)
                      .allocation_strategy(iox2::AllocationStrategy::PowerOfTwo)
                      .create()
                      .value();
    client_ = std::make_unique<ClientType>(std::move(client));
}

template <typename ResponseHandler>
auto ParameterRpcClient::call_remote(capnp::MessageBuilder& request_builder,
                                      ResponseHandler handler,
                                      int timeout_ms)
    -> decltype(handler(std::declval<tcnart_msgs::rpc::ParameterRpcResponse::Reader>())) {
    using RetType = decltype(handler(std::declval<tcnart_msgs::rpc::ParameterRpcResponse::Reader>()));

    // Serialize request
    auto words = capnp::messageToFlatArray(request_builder);
    auto bytes = words.asBytes();

    // Loan and fill request slice
    auto loan = client_->loan_slice_uninit(bytes.size());
    if (!loan.has_value()) return RetType{};
    auto& request_uninit = loan.value();
    auto dst = request_uninit.payload_mut().data();
    std::memcpy(dst, bytes.begin(), bytes.size());
    auto request_init = iox2::assume_init(std::move(request_uninit));
    auto maybe_pending = iox2::send(std::move(request_init));
    if (!maybe_pending.has_value()) return RetType{};
    auto pending = std::move(maybe_pending.value());

    // Wait for response with timeout
    auto start = std::chrono::steady_clock::now();
    while (true) {
        auto maybe_response = pending.receive();
        if (maybe_response.has_value() && maybe_response.value().has_value()) {
            auto& response = maybe_response.value().value();
            auto response_slice = response.payload();
            auto response_data = response_slice.data();
            auto response_len = response_slice.number_of_elements();

            auto resp_words = kj::arrayPtr(
                reinterpret_cast<const capnp::word*>(response_data),
                response_len / sizeof(capnp::word));
            capnp::FlatArrayMessageReader reader(resp_words);
            auto resp = reader.getRoot<tcnart_msgs::rpc::ParameterRpcResponse>();
            return handler(resp);
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        if (elapsed_ms >= timeout_ms) break;

        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return RetType{};
}

std::vector<std::string> ParameterRpcClient::list_components(int timeout_ms) {
    capnp::MallocMessageBuilder builder;
    auto req = builder.initRoot<tcnart_msgs::rpc::ParameterRpcRequest>();
    req.setCommandType(tcnart_msgs::rpc::RPCCommandType::RPC_COMMAND_LIST_ENTITIES);

    return call_remote(builder, [](tcnart_msgs::rpc::ParameterRpcResponse::Reader resp) {
        std::vector<std::string> result;
        if (resp.which() == tcnart_msgs::rpc::ParameterRpcResponse::ENTITIES_LIST) {
            for (auto name : resp.getEntitiesList()) {
                result.emplace_back(name);
            }
        }
        return result;
    }, timeout_ms);
}

std::vector<ParameterSchemaEntry> ParameterRpcClient::get_parameter_schema(
    const std::string& entity_name, int timeout_ms) {
    capnp::MallocMessageBuilder builder;
    auto req = builder.initRoot<tcnart_msgs::rpc::ParameterRpcRequest>();
    req.setCommandType(tcnart_msgs::rpc::RPCCommandType::RPC_COMMAND_GET_PARAMETER_SCHEMA);
    req.setEntityName(entity_name);

    return call_remote(builder, [](tcnart_msgs::rpc::ParameterRpcResponse::Reader resp) {
        std::vector<ParameterSchemaEntry> result;
        if (resp.which() == tcnart_msgs::rpc::ParameterRpcResponse::PARAMETER_SCHEMA) {
            for (auto param : resp.getParameterSchema().getParameters()) {
                result.push_back({
                    param.getKey(),
                    param.getValueType(),
                });
            }
        }
        return result;
    }, timeout_ms);
}

std::vector<ParameterEntry> ParameterRpcClient::get_parameter_values(
    const std::string& entity_name, int timeout_ms) {
    capnp::MallocMessageBuilder builder;
    auto req = builder.initRoot<tcnart_msgs::rpc::ParameterRpcRequest>();
    req.setCommandType(tcnart_msgs::rpc::RPCCommandType::RPC_COMMAND_GET_PARAMETER_VALUES);
    req.setEntityName(entity_name);

    return call_remote(builder, [](tcnart_msgs::rpc::ParameterRpcResponse::Reader resp) {
        std::vector<ParameterEntry> result;
        if (resp.which() == tcnart_msgs::rpc::ParameterRpcResponse::PARAMETER_VALUES) {
            for (auto param : resp.getParameterValues().getParameters()) {
                result.push_back(read_parameter(param));
            }
        }
        return result;
    }, timeout_ms);
}

bool ParameterRpcClient::set_parameter_values(const std::string& entity_name,
                                                const std::vector<ParameterEntry>& values,
                                                int timeout_ms) {
    capnp::MallocMessageBuilder builder;
    auto req = builder.initRoot<tcnart_msgs::rpc::ParameterRpcRequest>();
    req.setCommandType(tcnart_msgs::rpc::RPCCommandType::RPC_COMMAND_SET_PARAMETER_VALUES);
    req.setEntityName(entity_name);
    auto payload = req.initPayload();
    auto params = payload.initParameters(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        params[i].setKey(values[i].key);
        set_parameter_value(params[i].initValue(), values[i]);
    }

    return call_remote(builder, [](tcnart_msgs::rpc::ParameterRpcResponse::Reader resp) {
        return resp.getResponseType() == tcnart_msgs::rpc::RPCResponseStatus::RPC_STATUS_SUCCESS;
    }, timeout_ms);
}

}  // namespace tcn::shm
