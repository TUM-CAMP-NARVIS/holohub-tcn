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

#include "zenoh_subscriber_op.hpp"
#include "cdr_decoder_op.hpp"

namespace {

/// Simple sink operator that logs received messages (test endpoint).
class LogSinkOp : public holoscan::Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(LogSinkOp)
    LogSinkOp() = default;

    void setup(holoscan::OperatorSpec& spec) override {
        spec.input<std::vector<uint8_t>>("input");
    }

    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext&,
                 holoscan::ExecutionContext&) override {
        auto data = op_input.receive<std::vector<uint8_t>>("input").value();

        count_++;
        if (count_ % 30 == 1) {
            HOLOSCAN_LOG_INFO("LogSink [{}]: received {} bytes (total: {})",
                              name(), data.size(), count_);
        }
    }

 private:
    uint64_t count_ = 0;
};

/// Pre-compose discovery data passed from main() to compose().
struct ZenohDiscoveryData {
    std::shared_ptr<zenoh::Session> session;
    std::string topic_prefix;
    std::string capture_node;
    std::vector<std::string> stream_topics;
};

}  // namespace

class TcnZenohReceiverApp : public holoscan::Application {
 public:
    explicit TcnZenohReceiverApp(std::shared_ptr<ZenohDiscoveryData> discovery)
        : discovery_(std::move(discovery)) {}

    void compose() override {
        using namespace holoscan;

        HOLOSCAN_LOG_INFO("Starting TCN Zenoh Receiver (C++)");

        auto& topics = discovery_->stream_topics;

        if (topics.empty()) {
            HOLOSCAN_LOG_WARN("No stream topics configured — subscribing to wildcard");
            // Subscribe to all streams under the capture node
            std::string wildcard_topic = discovery_->topic_prefix + "/" +
                                         discovery_->capture_node + "/stream/**";
            topics.push_back(wildcard_topic);
        }

        for (size_t i = 0; i < topics.size(); ++i) {
            auto& topic = topics[i];
            std::string stream_name = "stream_" + std::to_string(i);

            HOLOSCAN_LOG_INFO("Creating pipeline for topic: {}", topic);

            // Zenoh subscriber
            auto async_cond = make_condition<AsynchronousCondition>(
                stream_name + "_async_condition");
            auto subscriber_op = make_operator<tcn::ops::TcnZenohSubscriberOp>(
                "subscriber_" + stream_name,
                Arg("topic", topic),
                Arg("async_condition", async_cond));
            subscriber_op->set_session(discovery_->session);

            // CDR decoder
            auto decoder_op = make_operator<tcn::ops::TcnCdrDecoderOp>(
                "cdr_decoder_" + stream_name,
                Arg("source_name", stream_name),
                Arg("stream_index", static_cast<int32_t>(i)));

            // Log sink (test endpoint — replace with NvVideoDecoder + HolovizOp later)
            auto sink_op = make_operator<LogSinkOp>("sink_" + stream_name);

            add_flow(subscriber_op, decoder_op, {{"output", "input"}, {"type_name", "type_name"}});
            add_flow(decoder_op, sink_op, {{"output", "input"}});
        }
    }

 private:
    std::shared_ptr<ZenohDiscoveryData> discovery_;
};

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
                      << "  -c, --config <file>     Config YAML (default: tcn_zenoh_receiver.yaml)\n"
                      << "  -s, --scheduler <type>  Scheduler: greedy|event_based (default: event_based)\n"
                      << "  -l, --log-level <lvl>   Log level: warn|info|debug (default: info)\n"
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

    // Read Zenoh config from YAML
    auto discovery = std::make_shared<ZenohDiscoveryData>();
    std::string zenoh_config_file;

    try {
        YAML::Node cfg = YAML::LoadFile(config_file);
        if (cfg["zenoh"]) {
            auto zenoh_cfg = cfg["zenoh"];
            discovery->topic_prefix = zenoh_cfg["topic_prefix"].as<std::string>("tcn/loc/pcpd");
            discovery->capture_node = zenoh_cfg["capture_node"].as<std::string>("k4a_capture_multi");
            zenoh_config_file = zenoh_cfg["zenoh_config_file"].as<std::string>("");

            // Optional: explicit stream topics list
            if (zenoh_cfg["stream_topics"]) {
                for (auto& t : zenoh_cfg["stream_topics"]) {
                    discovery->stream_topics.push_back(t.as<std::string>());
                }
            }
        }
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_WARN("Could not read config ({}), using defaults", e.what());
        discovery->topic_prefix = "tcn/loc/pcpd";
        discovery->capture_node = "k4a_capture_multi";
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
    discovery->session = std::make_shared<zenoh::Session>(std::move(session));
    HOLOSCAN_LOG_INFO("Zenoh session opened");

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

    return 0;
}
