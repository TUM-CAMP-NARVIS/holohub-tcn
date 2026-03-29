/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <yaml-cpp/yaml.h>

#define ZENOHCXX_ZENOHC 1
#include <zenoh.hxx>

#include "zenoh_receiver_op.hpp"

namespace {

/// Simple sink operator that discards input (needed to consume unused outputs).
class DummySinkOp : public holoscan::Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(DummySinkOp)
    DummySinkOp() = default;

    void setup(holoscan::OperatorSpec& spec) override {
        spec.input<holoscan::gxf::Entity>("input");
    }

    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext&,
                 holoscan::ExecutionContext&) override {
        op_input.receive<holoscan::gxf::Entity>("input");
    }
};

/// Pre-compose discovery data passed from main() to compose().
struct ZenohDiscoveryData {
    std::shared_ptr<zenoh::Session> session;
    std::string topic_prefix;
    std::string capture_node;
    std::vector<std::string> stream_types;
    std::vector<tcn::ops::ZenohStreamConfig> stream_configs;
};

}  // namespace

class TcnZenohReceiverApp : public holoscan::Application {
 public:
    explicit TcnZenohReceiverApp(std::shared_ptr<ZenohDiscoveryData> discovery)
        : discovery_(std::move(discovery)) {}

    void compose() override {
        using namespace holoscan;

        HOLOSCAN_LOG_INFO("Starting TCN Zenoh Receiver (C++)");

        const auto& configs = discovery_->stream_configs;
        if (configs.empty()) {
            HOLOSCAN_LOG_ERROR("No streams discovered — cannot compose pipeline");
            return;
        }

        // Read configuration
        auto& yaml_cfg = config().yaml_nodes();
        int32_t cuda_device_id = 0;

        if (!yaml_cfg.empty()) {
            auto root = yaml_cfg[0];
            if (root["pipeline"]) {
                auto pipe = root["pipeline"];
                cuda_device_id = pipe["device_id"].as<int32_t>(0);
            }
        }

        // GPU allocator for ZenohReceiverOp tensor uploads
        auto device_memory_pool = make_resource<UnboundedAllocator>(
            "zenoh_receiver_allocator");

        auto cuda_stream_pool = make_resource<CudaStreamPool>(
            "cuda_stream_pool",
            Arg("dev_id", cuda_device_id),
            Arg("stream_flags", static_cast<uint32_t>(0)),
            Arg("stream_priority", static_cast<uint32_t>(0)),
            Arg("reserved_size", static_cast<uint32_t>(configs.size())),
            Arg("max_size", static_cast<uint32_t>(64)));

        // --- TcnZenohReceiverOp (composite: subscribe + CDR decode + GPU upload) ---
        auto async_cond = make_condition<AsynchronousCondition>("zenoh_receiver_async");

        auto receiver_op = std::make_shared<tcn::ops::TcnZenohReceiverOp>();
        receiver_op->set_stream_configs(configs);
        receiver_op->set_session(discovery_->session);
        receiver_op->name("zenoh_receiver");
        receiver_op->fragment(this);
        receiver_op->init_spec();
        receiver_op->add_arg(Arg("async_condition", async_cond));
        receiver_op->add_arg(Arg("allocator", device_memory_pool));
        receiver_op->add_arg(Arg("cuda_stream_pool", cuda_stream_pool));

        // --- Per-stream output routing ---
        // Route each stream output to a dummy sink (SHM sender removed for now).
        for (const auto& cfg : configs) {
            auto sink = make_operator<DummySinkOp>("sink_" + cfg.name);
            add_flow(receiver_op, sink, {{cfg.name, "input"}});
        }
    }

 private:
    std::shared_ptr<ZenohDiscoveryData> discovery_;
};

// ---------------------------------------------------------------------------
// Pre-compose: Zenoh discovery (runs before the Holoscan runtime starts)
// ---------------------------------------------------------------------------
static std::shared_ptr<ZenohDiscoveryData> discover_zenoh(
    std::shared_ptr<zenoh::Session> session,
    const std::string& topic_prefix,
    const std::string& capture_node,
    const std::vector<std::string>& stream_types) {

    auto discovery = std::make_shared<ZenohDiscoveryData>();
    discovery->session = session;
    discovery->topic_prefix = topic_prefix;
    discovery->capture_node = capture_node;
    discovery->stream_types = stream_types;

    HOLOSCAN_LOG_INFO("Discovering Zenoh streams: prefix={}, node={}", topic_prefix, capture_node);

    discovery->stream_configs = tcn::ops::TcnZenohReceiverOp::discover_streams(
        *session, topic_prefix, capture_node, stream_types);

    if (discovery->stream_configs.empty()) {
        HOLOSCAN_LOG_ERROR("No streams discovered via Zenoh");
        return nullptr;
    }

    HOLOSCAN_LOG_INFO("Discovered {} streams", discovery->stream_configs.size());
    for (const auto& cfg : discovery->stream_configs) {
        HOLOSCAN_LOG_INFO("  {} -> {} ({}x{}, compression={})",
                          cfg.name, cfg.topic, cfg.image_width, cfg.image_height,
                          cfg.image_compression);
    }

    return discovery;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string config_file;
    std::string scheduler_type = "event_based";
    std::string log_level = "info";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_file = argv[++i];
        } else if ((arg == "-s" || arg == "--scheduler") && i + 1 < argc) {
            scheduler_type = argv[++i];
        } else if ((arg == "-l" || arg == "--log-level") && i + 1 < argc) {
            log_level = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "ARTEKMED Holoscan Zenoh Receiver (C++)\n"
                      << "Usage: " << argv[0] << " [options]\n"
                      << "  -c, --config <file>     Config YAML\n"
                      << "  -s, --scheduler <type>  Scheduler: greedy|event_based (default: event_based)\n"
                      << "  -l, --log-level <lvl>   Log level: warn|info|debug|trace (default: info)\n"
                      << "  -h, --help              Show this help\n";
            return 0;
        }
    }

    // Set log level
    if (log_level == "debug") {
        holoscan::set_log_level(holoscan::LogLevel::DEBUG);
    } else if (log_level == "info") {
        holoscan::set_log_level(holoscan::LogLevel::INFO);
    } else if (log_level == "warn") {
        holoscan::set_log_level(holoscan::LogLevel::WARN);
    } else if (log_level == "trace") {
        holoscan::set_log_level(holoscan::LogLevel::TRACE);
    }

    // Default config file path
    if (config_file.empty()) {
        std::string exe_path = argv[0];
        auto last_slash = exe_path.rfind('/');
        if (last_slash != std::string::npos) {
            config_file = exe_path.substr(0, last_slash + 1) + "tcn_zenoh_receiver.yaml";
        } else {
            config_file = "tcn_zenoh_receiver.yaml";
        }
    }

    // Read configuration
    std::string topic_prefix = "tcn/loc/pcpd";
    std::string capture_node = "k4a_capture_multi";
    std::string zenoh_config_file;
    std::vector<std::string> stream_types = {"color", "depth"};

    try {
        YAML::Node cfg = YAML::LoadFile(config_file);
        if (cfg["zenoh"]) {
            auto zenoh_cfg = cfg["zenoh"];
            topic_prefix = zenoh_cfg["topic_prefix"].as<std::string>(topic_prefix);
            capture_node = zenoh_cfg["capture_node"].as<std::string>(capture_node);
            zenoh_config_file = zenoh_cfg["zenoh_config_file"].as<std::string>("");

            if (zenoh_cfg["stream_types"]) {
                stream_types.clear();
                for (const auto& t : zenoh_cfg["stream_types"]) {
                    stream_types.push_back(t.as<std::string>());
                }
            }
        }
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_WARN("Could not read config ({}), using defaults", e.what());
    }

    // Open Zenoh session
    HOLOSCAN_LOG_INFO("Opening Zenoh session...");
    zenoh::Config zenoh_config = zenoh::Config::create_default();
    if (!zenoh_config_file.empty()) {
        try {
            zenoh_config = zenoh::Config::from_file(zenoh_config_file);
        } catch (const std::exception& e) {
            HOLOSCAN_LOG_WARN("Could not read Zenoh config file: {} ({})",
                              zenoh_config_file, e.what());
        }
    }

    auto session = zenoh::Session::open(std::move(zenoh_config));
    auto session_ptr = std::make_shared<zenoh::Session>(std::move(session));
    HOLOSCAN_LOG_INFO("Zenoh session opened");

    // Pre-compose: discover Zenoh streams
    auto discovery = discover_zenoh(session_ptr, topic_prefix, capture_node, stream_types);
    if (!discovery) {
        HOLOSCAN_LOG_ERROR("Zenoh discovery failed — no streams found. Exiting.");
        return 1;
    }

    // Create and configure application
    auto app = holoscan::make_application<TcnZenohReceiverApp>(discovery);
    app->config(config_file);

    // Configure scheduler
    if (scheduler_type == "greedy") {
        app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>(
            "gs", holoscan::Arg("stop_on_deadlock", true)));
    } else if (scheduler_type == "event_based") {
        app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
            "ebs", holoscan::Arg("worker_thread_number", static_cast<int64_t>(8))));
    } else {
        HOLOSCAN_LOG_ERROR("Invalid scheduler type: {}", scheduler_type);
        return 1;
    }

    // Run
    try {
        app->run();
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
        return 1;
    }

    // Clean up Zenoh session
    discovery->session.reset();
    session_ptr.reset();

    return 0;
}
