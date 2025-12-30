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

#ifndef UTILS_H
#define UTILS_H
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>
#include <cuda_runtime.h>

/**
 * CUDA driver API error check helper
 */
#define CudaCheck(FUNC)                                                                     \
  {                                                                                         \
    const CUresult result = FUNC;                                                           \
    if (result != CUDA_SUCCESS) {                                                           \
      const char *error_name = "";                                                          \
      cuGetErrorName(result, &error_name);                                                  \
      const char *error_string = "";                                                        \
      cuGetErrorString(result, &error_string);                                              \
      std::stringstream buf;                                                                \
      buf << "[" << __FILE__ << ":" << __LINE__ << "] CUDA driver error " << result << " (" \
          << error_name << "): " << error_string;                                           \
      throw std::runtime_error(buf.str().c_str());                                          \
    }                                                                                       \
  }
#define CUDA_TRY(stmt)                                                                       \
  {                                                                                          \
    cudaError_t cuda_status = stmt;                                                          \
    if (cudaSuccess != cuda_status) {                                                        \
      HOLOSCAN_LOG_ERROR("CUDA runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                              \
                         __LINE__,                                                           \
                         __FILE__,                                                           \
                         cudaGetErrorString(cuda_status),                                    \
                         int(cuda_status));                                                  \
      throw std::runtime_error("CUDA runtime call failed");                                  \
    }                                                                                        \
  }

namespace tcn {

template <auto A, typename...> auto value = A;

template<typename T>
struct tensor_primitive_type_map : std::false_type {
  using type = T;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kCustom;
};

template<>
struct tensor_primitive_type_map<int8_t> : std::true_type {
  using type = int8_t;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kInt8;
  constexpr static int8_t default_value{0};
};

template<>
struct tensor_primitive_type_map<uint8_t> : std::true_type {
  using type = uint8_t;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kUnsigned8;
  constexpr static uint8_t default_value{0};
};

template<>
struct tensor_primitive_type_map<int16_t> : std::true_type {
  using type = int16_t;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kInt16;
  constexpr static int16_t default_value{0};
};

template<>
struct tensor_primitive_type_map<uint16_t> : std::true_type {
  using type = uint16_t;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kUnsigned16;
  constexpr static uint16_t default_value{0};
};

template<>
struct tensor_primitive_type_map<int32_t> : std::true_type {
  using type = int32_t;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kInt32;
  constexpr static int32_t default_value{0};
};

template<>
struct tensor_primitive_type_map<uint32_t> : std::true_type {
  using type = uint32_t;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kUnsigned32;
  constexpr static uint32_t default_value{0};
};

template<>
struct tensor_primitive_type_map<int64_t> : std::true_type {
  using type = int64_t;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kInt64;
  constexpr static int64_t default_value{0};
};

template<>
struct tensor_primitive_type_map<uint64_t> : std::true_type {
  using type = uint64_t;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kUnsigned64;
  constexpr static uint64_t default_value{0};
};

// template<>
// struct tensor_primitive_type_map<??> : std::true_type {
//   using type = ??;
//   constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kFloat16;
// };

template<>
struct tensor_primitive_type_map<float> : std::true_type {
  using type = float;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kFloat32;
  constexpr static float default_value{0.f};
};

template<>
struct tensor_primitive_type_map<double> : std::true_type {
  using type = double;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kFloat64;
  constexpr static double default_value{0};
};

template<>
struct tensor_primitive_type_map<std::complex<double>> : std::true_type {
  using type = std::complex<double>;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kComplex64;
  constexpr static std::complex<double> default_value{0, 0};
};

template<>
struct tensor_primitive_type_map<std::complex<long double>> : std::true_type {
  using type = std::complex<long double>;
  constexpr static auto tensor_type = nvidia::gxf::PrimitiveType::kComplex128;
  constexpr static std::complex<long double> default_value{0, 0};
};




template<typename T>
bool allocate_named_tensor(
    nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator,
    cudaStream_t stream,
    nvidia::gxf::Entity& entity,
    const nvidia::gxf::Shape tensor_shape,
    const nvidia::gxf::MemoryStorageType tensor_storage_type,
    const std::string& out_tensor_name,
    nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor,
    bool initialize_buffer=false) {

  if constexpr (!tensor_primitive_type_map<T>::value) {
    static_assert(value<false, T>, "Unsupported primitive type..");
  }

  auto maybe_tensor = entity.add<nvidia::gxf::Tensor>(out_tensor_name.c_str());
  if (!maybe_tensor) {
    HOLOSCAN_LOG_ERROR("Failed to allocate {} tensor for entity.", out_tensor_name);
    return false;
  }
  tensor = maybe_tensor.value();
  constexpr auto tensor_dtype = tensor_primitive_type_map<T>::tensor_type;
  const uint64_t tensor_bytes_per_element = nvidia::gxf::PrimitiveTypeSize(tensor_dtype);
  auto tensor_strides =
      nvidia::gxf::ComputeTrivialStrides(tensor_shape, tensor_bytes_per_element);

  auto tensor_result = tensor->reshapeCustom(tensor_shape,
                                                   tensor_dtype,
                                                   tensor_bytes_per_element,
                                                   tensor_strides,
                                                   tensor_storage_type,
                                                   allocator);
  if (!tensor_result) {
    HOLOSCAN_LOG_ERROR("failed to allocate tensor");
    return false;
  }
  if (initialize_buffer) {
    // write zero's on init
    if (auto maybe_data = tensor->data<uint8_t>()) {
      uint8_t* gpu_data = maybe_data.value();
      size_t history_bytes = tensor->size() * tensor_bytes_per_element;
      HOLOSCAN_CUDA_CALL(cudaMemsetAsync(gpu_data, tensor_primitive_type_map<T>::default_value, history_bytes, stream));
    }
  }
  return true;
}


template<typename T>
bool allocate_tensor(
    nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator,
    cudaStream_t stream,
    const nvidia::gxf::Shape tensor_shape,
    const nvidia::gxf::MemoryStorageType tensor_storage_type,
    std::shared_ptr<nvidia::gxf::Tensor>& tensor,
    bool initialize_buffer = false) {

  if constexpr (!tensor_primitive_type_map<T>::value) {
    static_assert(value<false, T>, "Unsupported primitive type..");
  }

  tensor = std::make_shared<nvidia::gxf::Tensor>();
  constexpr auto tensor_dtype = tensor_primitive_type_map<T>::tensor_type;
  const uint64_t tensor_bytes_per_element = nvidia::gxf::PrimitiveTypeSize(tensor_dtype);
  auto tensor_strides =
      nvidia::gxf::ComputeTrivialStrides(tensor_shape, tensor_bytes_per_element);

  auto tensor_result = tensor->reshapeCustom(tensor_shape,
                                                   tensor_dtype,
                                                   tensor_bytes_per_element,
                                                   tensor_strides,
                                                   tensor_storage_type,
                                                   allocator);
  if (!tensor_result) {
    HOLOSCAN_LOG_ERROR("failed to allocate tensor");
    return false;
  }

  if (initialize_buffer) {
    // write zero's on init
    if (auto maybe_data = tensor->data<uint8_t>()) {
      uint8_t* gpu_data = maybe_data.value();
      size_t history_bytes = tensor->size() * tensor_bytes_per_element;
      HOLOSCAN_CUDA_CALL(cudaMemsetAsync(gpu_data, tensor_primitive_type_map<T>::default_value, history_bytes, stream));
    }
  }
  return true;
}

}


#endif /* UTILS_H */
