// TemporalFilterKernel.cu
#include <cuda_runtime.h>
#include <cmath>
#include "tcn_depthimage_temporal_filter_kernel.cuh"
#include "detail/processing_algorithms.cuh"

__global__ void temporal_filtering_u16_kernel(TemporalFilterParams bp) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= bp.width || y >= bp.height) return;
  const int idx = y * bp.width + x;

  uint16_t currentVal = bp.depth[idx];
  uint16_t prevVal = bp.temporalFilterLastFrame[idx];

  if (currentVal) {
    if (!prevVal || __isnan(prevVal)) {
      bp.temporalFilterLastFrame[idx] = currentVal;
      bp.temporalFilterHistory[idx] = bp.temporalFilterMask;
      bp.temporalFilterOutput[idx] = currentVal;
    } else {
      auto difference =
          static_cast<uint16_t>(abs(currentVal - prevVal));
      if (difference < bp.temporalFilterDelta) {
        //Difference is smaller than our delta.. average the last two frames
        uint8_t his = bp.temporalFilterHistory[idx];
        his |= bp.temporalFilterMask;
        bp.temporalFilterHistory[idx] = his;
        const float filtered =
            bp.temporalFilterAlpha * static_cast<float>(currentVal) + bp.temporalFilterOneMinusAlpha * static_cast<float>(prevVal);
        const auto result = static_cast<uint16_t>(filtered);
        bp.temporalFilterOutput[idx] = result;
        bp.temporalFilterLastFrame[idx] = result;
      } else {
        bp.temporalFilterOutput[idx] = currentVal;
        bp.temporalFilterLastFrame[idx] = currentVal;
        bp.temporalFilterHistory[idx] = bp.temporalFilterMask;
      }
    }
  } else {
    uint8_t hist = bp.temporalFilterHistory[idx];
    if (prevVal) {
      uint8_t classification = bp.temporalFilterPersistenceMap[hist];
      if (classification & hist) {
        bp.temporalFilterOutput[idx] = prevVal;
      } else {
        bp.temporalFilterOutput[idx] = 0;
      }
    } else {
      //Write 0
      bp.temporalFilterOutput[idx] = 0;
    }
    hist &= bp.temporalFilterMask;
    bp.temporalFilterHistory[idx] = hist;
  }


}