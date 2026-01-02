#include <cuda.h>
#include <cuda_runtime.h>
#include "../common/utils.h"

#include "../cuda/tcn_depthimage_temporal_filter_kernel.cuh"
#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_depthimage_temporal_filter.cuh"

#include "../common/datatypes.hpp"

#include <gxf/std/tensor.hpp>

namespace tcn::ops {

void TcnDepthImageTemporalFilterOp::setup(holoscan::OperatorSpec& spec) {
  using holoscan::Arg;
  HOLOSCAN_LOG_DEBUG("TcnDepthImageTemporalFilterOp::setup");

  // Inputs
  spec.input<holoscan::gxf::Entity>("input");  // device, [H, W], uint16

  // Outputs (planar)
  spec.output<holoscan::gxf::Entity>("output");

  // Configurable params (optional)
  spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");
  spec.param(persistence_, "persistence", "Temporal Filter Persistence", "", static_cast<uint8_t>(8));
  spec.param(delta_, "delta", "Temporal Filter Delta", "", static_cast<uint16_t>(30));
  spec.param(alpha_, "alpha", "Temporal Filter Alpha", "", 0.15f);
  spec.param(in_tensor_name_, "in_tensor_name", "Input Tensor Name", "", ""s);
  spec.param(out_tensor_name_, "out_tensor_name", "Output Tensor Name", "", ""s);

  spec.param(cuda_device_ordinal_,
             "cuda_device_ordinal",
             "CudaDeviceOrdinal",
             "Device to use for CUDA operations",
             holoscan::ParameterFlag::kOptional);
  // cuda_stream_handler_.define_params(spec);
}


void TcnDepthImageTemporalFilterOp::initialize() {
  HOLOSCAN_LOG_DEBUG("TcnDepthImageTemporalFilterOp::initialize");

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

void TcnDepthImageTemporalFilterOp::start() {
  // Initialize CUDA
  CudaCheck(cuInit(0));

  // Get the CUDA device
  CUdevice cu_device;
  CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal_.get()));
  cu_device_ = cu_device;

  // Retain the primary context for the device
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));
}


void TcnDepthImageTemporalFilterOp::compute(holoscan::InputContext& op_input,
                                            holoscan::OutputContext& op_output,
                                            holoscan::ExecutionContext& context) {


  // Receive tensors
  auto maybe_depth_t_entity = op_input.receive<holoscan::gxf::Entity>("input");
  if (!maybe_depth_t_entity) {
    throw std::runtime_error("Failed to read input entity");
  }

  auto depth_t = maybe_depth_t_entity.value().get<holoscan::Tensor>(in_tensor_name_.get().c_str());
  cudaStream_t cuda_stream = op_input.receive_cuda_stream("input", true, false);

  const auto& depth_shape = depth_t->shape();  // [H,W]
  const int H = static_cast<int>(depth_shape[0]);
  const int W = static_cast<int>(depth_shape[1]);

  // Map tensors to raw device pointers
  auto* depth_ptr = static_cast<uint16_t*>(depth_t->data());

  // Allocate Holoscan outputs (device)

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

  uint16_t* filtered_image_ptr = nullptr;
  nvidia::gxf::Handle<nvidia::gxf::Tensor> filtered_image = nullptr;

  // temporary buffers for temporal filtering
  uint16_t* temporal_filter_last_frame_ptr = nullptr;
  uint8_t* temporal_filter_history_ptr = nullptr;
  uint8_t* temporal_filter_persistence_map_ptr = nullptr;

  // create output
  auto maybe_filtered_image_entity = nvidia::gxf::Entity::New(context.context());
  if (!maybe_filtered_image_entity) {
    throw std::runtime_error("Failed to allocate message for output filtered_image tensor.");
  }
  nvidia::gxf::Entity filtered_image_entity = std::move(maybe_filtered_image_entity.value());

  if (!tcn::allocate_named_tensor<uint16_t>(
    allocator.value(),
    cuda_stream,
    filtered_image_entity,
    nvidia::gxf::Shape{{H, W, 1}},
    nvidia::gxf::MemoryStorageType::kDevice,
    out_tensor_name_.get(),
    filtered_image
    )) {
    throw std::runtime_error("Failed to allocate message for filtered_image.");
  }

  if (auto maybe_filtered_image_data = filtered_image->data<uint16_t>()) {
    filtered_image_ptr = maybe_filtered_image_data.value();
  } else {
    HOLOSCAN_LOG_ERROR("error access filtered_image tensor data");
    return;
  }

  // create / use last_frame device buffer
  if (!buffer_last_frame_) {
    if (!tcn::allocate_tensor<uint16_t>(
      allocator.value(),
      cuda_stream,
      nvidia::gxf::Shape{{H, W, 1}},
      nvidia::gxf::MemoryStorageType::kDevice,
      buffer_last_frame_,
      true
      )) {
      throw std::runtime_error("Failed to allocate message for last_frame.");
    }
  }
  if (auto maybe_last_frame_data = buffer_last_frame_->data<uint16_t>()) {
    temporal_filter_last_frame_ptr = maybe_last_frame_data.value();
  } else {
    HOLOSCAN_LOG_ERROR("error access last_frame tensor data");
    return;
  }

  // create / use history
  if (!buffer_history_) {
    if (!tcn::allocate_tensor<uint8_t>(
      allocator.value(),
      cuda_stream,
      nvidia::gxf::Shape{{H, W, 1}},
      nvidia::gxf::MemoryStorageType::kDevice,
      buffer_history_,
      true
      )) {
      throw std::runtime_error("Failed to allocate message for history.");
    }
  }
  if (auto maybe_history_data = buffer_history_->data<uint8_t>()) {
    temporal_filter_history_ptr = maybe_history_data.value();
  } else {
    HOLOSCAN_LOG_ERROR("error access history tensor data");
    return;
  }

  // persistence map
  if (!buffer_persistence_map_) {
    buildPersistenceMap(allocator.value(), cuda_stream);
  }
  if (auto maybe_persistence_map_data = buffer_persistence_map_->data<uint8_t>()) {
    temporal_filter_persistence_map_ptr = maybe_persistence_map_data.value();
  } else {
    HOLOSCAN_LOG_ERROR("error access persistence_map tensor data");
    return;
  }

  // Launch kernel
  TemporalFilterParams params{};
  params.depth = depth_ptr;
  params.temporalFilterLastFrame = temporal_filter_last_frame_ptr;
  params.temporalFilterHistory = temporal_filter_history_ptr;
  params.temporalFilterPersistenceMap = temporal_filter_persistence_map_ptr;
  params.temporalFilterOutput = filtered_image_ptr;

  params.width = W;
  params.height = H;
  params.temporalFilterDelta = delta_.get();
  params.temporalFilterAlpha = alpha_.get();
  params.temporalFilterOneMinusAlpha = 1.0f - params.temporalFilterAlpha;

  params.temporalFilterMask = 0x01 << current_frame_index_;

  const dim3 block(16, 16);
  const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
  temporal_filtering_u16_kernel<<<grid, block, 0, cuda_stream>>>(params);

  auto filtered_image_message = holoscan::gxf::Entity(std::move(filtered_image_entity));
  // op_output.set_cuda_stream(cuda_stream, "output");
  op_output.emit(filtered_image_message, "output");

  current_frame_index_ = (current_frame_index_+1)%8;
}

void TcnDepthImageTemporalFilterOp::buildPersistenceMap(nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator, cudaStream_t stream)
{
  std::vector<uint8_t> persistence_map(PERSISTENCE_MAP_SIZE);
  uint8_t* host_data = persistence_map.data();
  for(uint16_t i = 0; i < PERSISTENCE_MAP_SIZE;++i) {
    const uint8_t last_7 = (i & 1) != 0;
    const uint8_t last_6 = (i & 2) != 0;
    const uint8_t last_5 = (i & 4) != 0;
    const uint8_t last_4 = (i & 8) != 0;
    const uint8_t last_3 = (i & 16) != 0;
    const uint8_t last_2 = (i & 32) != 0;
    const uint8_t last_1 = (i & 64) != 0;
    const uint8_t last_frame = (i & 128) != 0;
    host_data[i] = 0;
    switch(persistence_.get()){
        case 1:
            if(last_frame+last_1+last_2+last_3+last_4+last_5+last_6+last_7 >= 8) {
                host_data[i] = 1;
            }
            break;
        case 2:
            if(last_frame+last_1+last_2>=2) {
                host_data[i] = 1;
            }
            break;
        case 3:
            if(last_frame+last_1+last_2+last_3>= 2) {
                host_data[i] = 1;
            }
            break;
        case 4:
            if(last_frame+last_1+last_2+last_3+last_4+last_5+last_6+last_7 >= 2) {
                host_data[i] = 1;
            }
            break;
        case 5:
            if(last_frame+last_1>= 1) {
                host_data[i] = 1;
            }
            break;
        case 6:
            if(last_frame+last_1+last_2+last_3+last_4>= 1) {
                host_data[i] = 1;
            }
            break;
        case 7:
            if(last_frame+last_1+last_2+last_3+last_4+last_5+last_6+last_7 >= 1) {
                host_data[i] = 1;
            }
            break;
        case 8:
            host_data[i] = 1;
            break;
        default:
          HOLOSCAN_LOG_ERROR("unknown persistence_map value");
          return;
    }
  }

  // allocate gpu buffer
  if (!tcn::allocate_tensor<uint8_t>(
        allocator,
        stream,
        nvidia::gxf::Shape{{PERSISTENCE_MAP_SIZE}},
        nvidia::gxf::MemoryStorageType::kDevice,
        buffer_persistence_map_
        )) {
    throw std::runtime_error("Failed to allocate message for persistence_mape.");
  }
  if (auto maybe_gpu_data = buffer_persistence_map_->data<uint8_t>()) {
    // upload data
    HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(maybe_gpu_data.value(), persistence_map.data(), PERSISTENCE_MAP_SIZE,
                                cudaMemcpyHostToDevice, stream));
    return;
  }
  HOLOSCAN_LOG_ERROR("GPU tensor allocation for persistence_map failed");
}

}  // namespace tcn::ops