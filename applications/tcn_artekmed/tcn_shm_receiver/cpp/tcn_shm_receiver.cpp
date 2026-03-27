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

#include <cmath>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "iox2/iceoryx2.hpp"

// TCN operators (headers exposed via BUILD_INTERFACE from each operator target)
#include "shm_subscriber_op.hpp"
#include "shm_synchronized_buffer_receiver.hpp"
#include "device_context_service.hpp"
#include "xy_lookup_table_source_op.hpp"
#include "tcn_stream_splitter.hpp"
#include "tcn_stream_merger.hpp"
#include "tcn_flatten_tensor.cuh"
#include "tcn_depthimage_backprojection.cuh"
#include "tcn_depthimage_temporal_filter.cuh"
#include "tcn_depthimage_weights.cuh"
#include "tcn_texture_sampler.cuh"
#include "tcn_depthimage_max_distance.cuh"
#include "tcn_depthimage_fgbg_mask.cuh"
#include "tcn_depthimage_apply_mask.cuh"

namespace {

/// Simple sink operator that discards its input (needed to consume unused outputs).
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

}  // namespace

/// Pre-compose discovery data passed from main() to compose().
struct ShmDiscoveryData {
    std::shared_ptr<tcn::shm::ShmSynchronizedBufferReceiver> receiver;
    std::shared_ptr<tcn::ops::DeviceContextService> ctx_service;
    std::vector<std::string> camera_names;

    // Native channel info (parsed from Cap'n Proto while SHM sample was alive)
    std::vector<tcn::shm::ChannelPortInfo> depth_channels;
    std::vector<tcn::shm::ChannelPortInfo> color_channels;
    size_t max_frame_size = 0;
    size_t num_channels = 0;
};

class TcnShmReceiverApp : public holoscan::Application {
 public:
    explicit TcnShmReceiverApp(std::shared_ptr<ShmDiscoveryData> discovery)
        : discovery_(std::move(discovery)) {}

    void compose() override {
        using namespace holoscan;

        HOLOSCAN_LOG_INFO("Starting TCN Shm Receiver (C++)");

        // Read configuration
        auto camera_cfg = from_config("camera_stream_processing");
        int32_t cuda_device_id = 0;
        int32_t block_memory_buffer_size = 8;
        bool enable_temporal_filter = false;
        bool enable_background_subtract = false;
        bool enable_compute_weights = false;
        bool enable_warp_colorimage = false;

        // Parse YAML config manually since from_config returns ArgList
        auto& yaml_cfg = config().yaml_nodes();
        if (!yaml_cfg.empty()) {
            auto root = yaml_cfg[0];
            if (root["camera_stream_processing"]) {
                auto csp = root["camera_stream_processing"];
                cuda_device_id = csp["device_id"].as<int32_t>(0);
                block_memory_buffer_size = csp["buffer_size"].as<int32_t>(8);
                enable_temporal_filter = csp["enable_temporal_filter"].as<bool>(false);
                enable_background_subtract = csp["enable_background_substract"].as<bool>(false);
                enable_compute_weights = csp["enable_compute_weights"].as<bool>(false);
                enable_warp_colorimage = csp["enable_warp_colorimage"].as<bool>(false);
            }
        }

        auto shm_cfg = from_config("shared_memory");
        std::string shm_stream_name = "camera_streams";
        int32_t cycle_time_ms = 1;
        if (!yaml_cfg.empty()) {
            auto root = yaml_cfg[0];
            if (root["shared_memory"]) {
                auto sc = root["shared_memory"];
                shm_stream_name = sc["stream_name"].as<std::string>("camera_streams");
                cycle_time_ms = sc["cycle_time_ms"].as<int32_t>(1);
            }
        }

        bool enable_colorimage = true;
        bool enable_depthimage = false;
        bool enable_weights_view = false;
        bool enable_pointcloud = false;
        bool enable_warped_color_view = false;
        if (!yaml_cfg.empty()) {
            auto root = yaml_cfg[0];
            if (root["debug_output"]) {
                auto dbg = root["debug_output"];
                enable_colorimage = dbg["enable_colorimage"].as<bool>(true);
                enable_depthimage = dbg["enable_depthimage"].as<bool>(false);
                enable_weights_view = dbg["enable_weights"].as<bool>(false);
                enable_pointcloud = dbg["enable_pointcloud"].as<bool>(false);
                enable_warped_color_view = dbg["enable_warped_color"].as<bool>(false);
            }
        }

        size_t num_channels = discovery_->num_channels;

        // Create shared resources
        HOLOSCAN_LOG_INFO("Creating CUDA stream pool with {} reserved streams on device {}",
                          num_channels, cuda_device_id);
        auto cuda_stream_pool = make_resource<CudaStreamPool>(
            "cuda_stream_pool",
            Arg("dev_id", cuda_device_id),
            Arg("stream_flags", static_cast<uint32_t>(0)),
            Arg("stream_priority", static_cast<uint32_t>(0)),
            Arg("reserved_size", static_cast<uint32_t>(num_channels)),
            Arg("max_size", static_cast<uint32_t>(256)));

        HOLOSCAN_LOG_INFO("Creating device memory pool: {} bytes, {} blocks on device {}",
                          discovery_->max_frame_size,
                          num_channels * block_memory_buffer_size,
                          cuda_device_id);
        auto device_memory_pool = make_resource<BlockMemoryPool>(
            "shm_subscriber_device_pool",
            Arg("storage_type", static_cast<int32_t>(1)),  // DEVICE
            Arg("block_size", static_cast<int64_t>(discovery_->max_frame_size)),
            Arg("num_blocks", static_cast<int64_t>(num_channels * block_memory_buffer_size)),
            Arg("dev_id", cuda_device_id));

        // Create async condition for SHM subscriber
        auto async_cond = make_condition<AsynchronousCondition>("shm_async_condition");

        // SHM Subscriber
        HOLOSCAN_LOG_INFO("Creating SHM subscriber for stream: {}", shm_stream_name);
        auto subscriber_op = make_operator<tcn::ops::TcnShmSubscriberOp>(
            "shm_subscriber",
            Arg("stream_name", shm_stream_name),
            Arg("cycle_time_ms", cycle_time_ms),
            Arg("allocator", device_memory_pool),
            Arg("async_condition", async_cond));
        subscriber_op->set_receiver(discovery_->receiver);

        // Stream Splitter — routes depth entity to per-camera outputs
        std::vector<std::string> depth_channel_names;
        for (auto& ch : discovery_->depth_channels) {
            depth_channel_names.push_back(ch.name);
        }

        HOLOSCAN_LOG_INFO("Creating stream splitter for {} depth channels", depth_channel_names.size());
        auto split_op = make_operator<tcn::ops::TcnStreamSplitterOp>(
            "stream_splitter",
            Arg("channel_names", depth_channel_names),
            Arg("cuda_stream_pool", cuda_stream_pool));
        split_op->set_channel_names_init(depth_channel_names);
        add_flow(subscriber_op, split_op, {{"depth_outputs", "receivers"}});

        // --- HolovizOp visualizers (configured before per-camera loop) ---

        // Points visualizer
        std::shared_ptr<ops::HolovizOp> points_visualizer;
        if (enable_pointcloud) {
            HOLOSCAN_LOG_INFO("Creating pointcloud visualizer");
            ops::HolovizOp::InputSpec spec("positions", ops::HolovizOp::InputType::POINTS_3D);
            spec.color_ = {1.0f, 0.0f, 0.0f, 1.0f};
            points_visualizer = make_operator<ops::HolovizOp>(
                "points_visualizer",
                from_config("points_holoviz"),
                Arg("tensors", std::vector<ops::HolovizOp::InputSpec>{spec}),
                Arg("allocator", device_memory_pool),
                Arg("cuda_stream_pool", cuda_stream_pool));
        }

        // Weights visualizer (grid layout for multi-camera)
        std::shared_ptr<ops::HolovizOp> weights_visualizer;
        if (enable_weights_view) {
            HOLOSCAN_LOG_INFO("Creating weights visualizer");
            size_t num_weights = discovery_->camera_names.size();
            int grid_size = num_weights > 0
                ? static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_weights))))
                : 1;
            float tile_size = 1.0f / grid_size;

            std::vector<ops::HolovizOp::InputSpec> weights_specs;
            for (size_t i = 0; i < num_weights; ++i) {
                auto& cam = discovery_->camera_names[i];
                int row = static_cast<int>(i) / grid_size;
                int col = static_cast<int>(i) % grid_size;

                ops::HolovizOp::InputSpec spec(cam, ops::HolovizOp::InputType::COLOR);
                ops::HolovizOp::InputSpec::View view;
                view.offset_x_ = col * tile_size;
                view.offset_y_ = row * tile_size;
                view.width_ = tile_size;
                view.height_ = tile_size;
                spec.views_ = {view};
                weights_specs.push_back(spec);
            }

            weights_visualizer = make_operator<ops::HolovizOp>(
                "weights_visualizer",
                from_config("weights_holoviz"),
                Arg("tensors", weights_specs),
                Arg("allocator", device_memory_pool),
                Arg("cuda_stream_pool", cuda_stream_pool));
        }

        // Warped color visualizer (grid layout for multi-camera)
        std::shared_ptr<ops::HolovizOp> warped_color_visualizer;
        if (enable_warped_color_view) {
            HOLOSCAN_LOG_INFO("Creating warped color visualizer");
            size_t num_cameras = discovery_->camera_names.size();
            int grid_size = num_cameras > 0
                ? static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_cameras))))
                : 1;
            float tile_size = 1.0f / grid_size;

            std::vector<ops::HolovizOp::InputSpec> wc_specs;
            for (size_t i = 0; i < num_cameras; ++i) {
                auto& cam = discovery_->camera_names[i];
                int row = static_cast<int>(i) / grid_size;
                int col = static_cast<int>(i) % grid_size;

                ops::HolovizOp::InputSpec spec(cam, ops::HolovizOp::InputType::COLOR);
                ops::HolovizOp::InputSpec::View view;
                view.offset_x_ = col * tile_size;
                view.offset_y_ = row * tile_size;
                view.width_ = tile_size;
                view.height_ = tile_size;
                spec.views_ = {view};
                wc_specs.push_back(spec);
            }

            warped_color_visualizer = make_operator<ops::HolovizOp>(
                "warped_color_visualizer",
                from_config("warped_color_holoviz"),
                Arg("tensors", wc_specs),
                Arg("allocator", device_memory_pool),
                Arg("cuda_stream_pool", cuda_stream_pool));
        }

        // --- Per-camera processing pipelines ---
        std::vector<std::pair<std::shared_ptr<Operator>,
                              std::set<std::pair<std::string, std::string>>>>
            position_merge_connections;

        auto& ctx_service = discovery_->ctx_service;

        for (auto& channel : discovery_->depth_channels) {
            auto channel_name = channel.name;
            auto camera_name_opt = ctx_service->get_camera_name_from_port_name(channel_name);
            if (!camera_name_opt) {
                HOLOSCAN_LOG_WARN("Could not extract camera name from: {}", channel_name);
                continue;
            }
            auto camera_name = *camera_name_opt;
            auto color_model = ctx_service->get_color_camera_model(camera_name);

            std::shared_ptr<Operator> prev_op = split_op;
            std::string prev_output = channel_name;

            // Optional: Temporal filter
            if (enable_temporal_filter) {
                auto ditf_op = make_operator<tcn::ops::TcnDepthImageTemporalFilterOp>(
                    camera_name + "_temporal_filter",
                    from_config("depthimage_temporal_filter"),
                    Arg("allocator", device_memory_pool),
                    Arg("cuda_stream_pool", cuda_stream_pool),
                    Arg("in_tensor_name", std::string("")),
                    Arg("out_tensor_name", std::string("")),
                    Arg("cuda_device_ordinal", cuda_device_id));
                add_flow(split_op, ditf_op, {{channel_name, "input"}});
                prev_op = ditf_op;
                prev_output = "output";
            }

            // Optional: Background subtraction
            if (enable_background_subtract) {
                HOLOSCAN_LOG_INFO("Creating background subtraction: {}", camera_name);

                // MaxDistance: computes running max distance from depth
                auto dimd_op = make_operator<tcn::ops::TcnDepthImageMaxDistanceOp>(
                    camera_name + "_max_distance",
                    Arg("allocator", device_memory_pool),
                    Arg("cuda_stream_pool", cuda_stream_pool));
                add_flow(prev_op, dimd_op, {{prev_output, "input"}});

                // FgBg Mask: generates foreground mask from depth vs background
                auto difgbg_op = make_operator<tcn::ops::TcnDepthImageFgbgMaskOp>(
                    camera_name + "_fg_bg_mask",
                    from_config("depthimage_fgbg_mask"),
                    Arg("allocator", device_memory_pool),
                    Arg("enable_foreground", true),
                    Arg("enable_background", false),
                    Arg("cuda_stream_pool", cuda_stream_pool));
                add_flow(split_op, difgbg_op, {{channel_name, "depth_image"}});
                add_flow(dimd_op, difgbg_op, {{"output", "background_image"}});

                // Apply Mask: applies foreground mask to depth image
                auto diam_op = make_operator<tcn::ops::TcnDepthImageApplyMaskOp>(
                    camera_name + "_apply_mask",
                    from_config("depthimage_apply_mask"),
                    Arg("allocator", device_memory_pool),
                    Arg("cuda_stream_pool", cuda_stream_pool));
                add_flow(split_op, diam_op, {{channel_name, "depth_image"}});
                add_flow(difgbg_op, diam_op, {{"foreground_mask", "mask_image"}});

                prev_op = diam_op;
                prev_output = "output";
            }

            // XY Lookup Table Source (fires once via CountCondition)
            HOLOSCAN_LOG_INFO("Creating XY lookup table source: {}", camera_name);
            auto xylt_op = make_operator<tcn::ops::XYLookupTableSourceOp>(
                "xylt_loader_" + camera_name,
                make_condition<CountCondition>("count", Arg("count", static_cast<int64_t>(1))),
                Arg("camera_name", camera_name),
                Arg("allocator", device_memory_pool));
            xylt_op->set_device_context_service(ctx_service);

            // Backprojection
            HOLOSCAN_LOG_INFO("Creating backprojection: {}", camera_name);
            auto depth_extrinsics = ctx_service->get_depth_extrinsics(camera_name);
            auto depth_to_color = ctx_service->get_color_to_depth_inv(camera_name);
            auto color_params_model = ctx_service->get_color_camera_model(camera_name);

            auto bp_op = make_operator<tcn::ops::TcnDepthImageBackprojectionOp>(
                camera_name + "_backprojection",
                from_config("depthimage_backprojection"),
                Arg("allocator", device_memory_pool),
                Arg("cuda_stream_pool", cuda_stream_pool),
                Arg("color_image_width", static_cast<int32_t>(color_params_model ? color_params_model->dimensions.x : 0)),
                Arg("color_image_height", static_cast<int32_t>(color_params_model ? color_params_model->dimensions.y : 0)),
                Arg("in_tensor_name", std::string("")),
                Arg("out_tensor_name", std::string("output")),
                Arg("enable_positions", true),
                Arg("enable_texcoords", enable_warp_colorimage),
                Arg("enable_depth_float", false),
                Arg("cuda_device_ordinal", cuda_device_id));

            // Set camera parameters on the backprojection operator
            if (color_params_model) {
                bp_op->add_arg(Arg("color_params", *color_params_model));
            }
            if (depth_extrinsics) {
                bp_op->add_arg(Arg("depth_extrinsics", *depth_extrinsics));
            }
            if (depth_to_color) {
                bp_op->add_arg(Arg("depth_to_color", *depth_to_color));
            }

            add_flow(prev_op, bp_op, {{prev_output, "depth_image"}});
            add_flow(xylt_op, bp_op, {{"xy_table", "xy_table"}});

            position_merge_connections.push_back(
                {bp_op, {{std::string("positions"),
                          camera_name + "_positions"}}});

            // Optional: Compute weights
            if (enable_compute_weights) {
                HOLOSCAN_LOG_INFO("Creating compute weights: {}", camera_name);
                auto cp_op = make_operator<tcn::ops::TcnDepthImageWeightsOp>(
                    camera_name + "_weights",
                    from_config("depthimage_weights"),
                    Arg("allocator", device_memory_pool),
                    Arg("cuda_stream_pool", cuda_stream_pool),
                    Arg("in_tensor_name", std::string("")),
                    Arg("out_tensor_name", camera_name),
                    Arg("cuda_device_ordinal", cuda_device_id));

                add_flow(prev_op, cp_op, {{prev_output, "depth_image"}});
                add_flow(xylt_op, cp_op, {{"xy_table", "xy_table"}});

                if (weights_visualizer) {
                    add_flow(cp_op, weights_visualizer, {{"output", "receivers"}});
                } else {
                    auto sink_op = make_operator<DummySinkOp>(
                        camera_name + "_weights_sink");
                    add_flow(cp_op, sink_op, {{"output", "input"}});
                }
            }

            // Optional: Warp color image
            if (enable_warp_colorimage) {
                auto wci_op = make_operator<tcn::ops::TcnTextureSamplerOp>(
                    camera_name + "_warp_colorimage",
                    Arg("allocator", device_memory_pool),
                    Arg("cuda_stream_pool", cuda_stream_pool),
                    Arg("in_color_tensor_name", camera_name + "_colorimage"),
                    Arg("in_texcoord_tensor_name", std::string("output")),
                    Arg("out_tensor_name", camera_name),
                    Arg("cuda_device_ordinal", cuda_device_id));

                add_flow(bp_op, wci_op, {{"texcoords", "texcoords"}});
                add_flow(subscriber_op, wci_op, {{"color_outputs", "color_image"}});

                if (warped_color_visualizer) {
                    add_flow(wci_op, warped_color_visualizer, {{"output", "receivers"}});
                } else {
                    auto sink_op = make_operator<DummySinkOp>(
                        camera_name + "_warped_color_sink");
                    add_flow(wci_op, sink_op, {{"output", "input"}});
                }
            }
        }

        // Merge position streams
        std::vector<std::string> merge_input_names;
        for (auto& [op, conn] : position_merge_connections) {
            for (auto& [src, dst] : conn) {
                merge_input_names.push_back(dst);
            }
        }

        HOLOSCAN_LOG_INFO("Merging {} position streams", merge_input_names.size());
        auto position_merge_op = make_operator<tcn::ops::TcnStreamMergerOp>(
            "point_fusion",
            Arg("input_port_names", merge_input_names),
            Arg("output_message_name", std::string("positions")),
            Arg("input_message_name", std::string("output")),
            Arg("fuse_buffers", true),
            Arg("allocator", device_memory_pool),
            Arg("cuda_stream_pool", cuda_stream_pool));
        position_merge_op->set_input_port_names_init(merge_input_names);

        for (auto& [op, conn] : position_merge_connections) {
            add_flow(op, position_merge_op, conn);
        }

        // Flatten tensor
        auto flt_op = make_operator<tcn::ops::TcnFlattenTensorOp>(
            "flatten_pointcloud",
            Arg("message_name", std::string("positions")),
            Arg("allocator", device_memory_pool),
            Arg("cuda_stream_pool", cuda_stream_pool));
        add_flow(position_merge_op, flt_op, {{"output", "input"}});

        // Point cloud consumer — either HolovizOp or DummySink
        if (points_visualizer) {
            add_flow(flt_op, points_visualizer, {{"output", "receivers"}});
        } else {
            auto pc_sink = make_operator<DummySinkOp>("point_cloud_sink");
            add_flow(flt_op, pc_sink, {{"output", "input"}});
        }

        // Color image consumer
        if (enable_colorimage) {
            HOLOSCAN_LOG_INFO("Creating color visualizer");
            auto color_viz = make_operator<ops::HolovizOp>(
                "color_visualizer",
                from_config("color_holoviz"),
                Arg("allocator", device_memory_pool),
                Arg("cuda_stream_pool", cuda_stream_pool));
            add_flow(subscriber_op, color_viz, {{"color_outputs", "receivers"}});
            add_flow(subscriber_op, color_viz, {{"color_output_specs", "input_specs"}});
        } else {
            auto ci_sink = make_operator<DummySinkOp>("color_image_sink");
            add_flow(subscriber_op, ci_sink, {{"color_outputs", "input"}});
        }

        // Depth image visualizer
        if (enable_depthimage) {
            HOLOSCAN_LOG_INFO("Creating depth visualizer");
            auto depth_viz = make_operator<ops::HolovizOp>(
                "depth_visualizer",
                from_config("depth_holoviz"),
                Arg("allocator", device_memory_pool),
                Arg("cuda_stream_pool", cuda_stream_pool));
            add_flow(subscriber_op, depth_viz, {{"depth_outputs", "receivers"}});
            add_flow(subscriber_op, depth_viz, {{"depth_output_specs", "input_specs"}});
        }
    }

 private:
    std::shared_ptr<ShmDiscoveryData> discovery_;
};

// ---------------------------------------------------------------------------
// Pre-compose: SHM discovery (runs before the Holoscan runtime starts)
// ---------------------------------------------------------------------------
static std::shared_ptr<ShmDiscoveryData> discover_shm(const std::string& stream_name) {
    auto discovery = std::make_shared<ShmDiscoveryData>();

    // Create iceoryx2 node and move it into the receiver so the node's
    // lifetime is tied to the receiver (not this stack frame).
    auto node = iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().value();
    discovery->receiver = std::make_shared<tcn::shm::ShmSynchronizedBufferReceiver>(std::move(node));
    discovery->ctx_service = std::make_shared<tcn::ops::DeviceContextService>();

    // Discover cameras
    HOLOSCAN_LOG_INFO("Discovering cameras in shared memory...");
    discovery->camera_names = tcn::shm::ShmSynchronizedBufferReceiver::discover_devices();

    if (discovery->camera_names.empty()) {
        HOLOSCAN_LOG_ERROR("No cameras found in shared memory");
        return nullptr;
    }

    for (auto& name : discovery->camera_names) {
        HOLOSCAN_LOG_INFO("  Found camera: {}", name);
        auto info = discovery->receiver->retrieve_device_context(name);
        if (info.is_valid) {
            discovery->ctx_service->add_device_context(name, std::move(info));
        }
    }

    // Retrieve channel configuration (returned as native structs — the Cap'n
    // Proto data was parsed while the SHM sample was alive, then released).
    HOLOSCAN_LOG_INFO("Retrieving channel config for stream: {}", stream_name);
    auto ports = discovery->receiver->retrieve_channel_config(stream_name);

    for (auto& port : ports) {
        discovery->max_frame_size = std::max(discovery->max_frame_size,
                                             static_cast<size_t>(port.frame_size));
        if (port.port_type == "depthimage") {
            discovery->depth_channels.push_back(std::move(port));
        } else if (port.port_type == "colorimage") {
            discovery->color_channels.push_back(std::move(port));
        }
    }

    discovery->num_channels = discovery->depth_channels.size() + discovery->color_channels.size();

    HOLOSCAN_LOG_INFO("Discovered {} depth + {} color channels (max frame: {} bytes)",
                      discovery->depth_channels.size(),
                      discovery->color_channels.size(),
                      discovery->max_frame_size);

    return discovery;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Parse command line arguments
    std::string config_file;
    std::string scheduler_type = "greedy";
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
            std::cout << "ARTEKMED Holoscan SHM Receiver (C++)\n"
                      << "Usage: " << argv[0] << " [options]\n"
                      << "  -c, --config <file>     Config YAML (default: tcn_shm_receiver.yaml)\n"
                      << "  -s, --scheduler <type>  Scheduler: greedy|event_based (default: greedy)\n"
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
        // Try to find config relative to executable
        std::string exe_path = argv[0];
        auto last_slash = exe_path.rfind('/');
        if (last_slash != std::string::npos) {
            config_file = exe_path.substr(0, last_slash + 1) + "tcn_shm_receiver.yaml";
        } else {
            config_file = "tcn_shm_receiver.yaml";
        }
    }

    // Read stream name from config before discovery
    std::string stream_name = "camera_streams";
    try {
        YAML::Node cfg = YAML::LoadFile(config_file);
        if (cfg["shared_memory"] && cfg["shared_memory"]["stream_name"]) {
            stream_name = cfg["shared_memory"]["stream_name"].as<std::string>();
        }
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_WARN("Could not pre-read config ({}), using default stream name", e.what());
    }

    // Pre-compose: discover SHM cameras and channels
    auto discovery = discover_shm(stream_name);
    if (!discovery) {
        HOLOSCAN_LOG_ERROR("SHM discovery failed — no cameras found. Exiting.");
        return 1;
    }

    // Create and configure application
    auto app = holoscan::make_application<TcnShmReceiverApp>(discovery);
    app->config(config_file);

    // Configure scheduler
    if (scheduler_type == "greedy") {
        app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>(
            "gs", holoscan::Arg("stop_on_deadlock", true)));
    } else if (scheduler_type == "event_based") {
        app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
            "ebs", holoscan::Arg("worker_thread_number", static_cast<int64_t>(24))));
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

    return 0;
}
