// TemporalFilterKernel.cuh
#pragma once

#include "../common/datatypes.hpp"
#include "../common/processing_algorithms.cuh"

static constexpr size_t PERSISTENCE_MAP_SIZE = 256;

struct TemporalFilterParams {
  const uint16_t* depth;    // [H*W]

  // temporary
  uint16_t* temporalFilterLastFrame;
  uint8_t* temporalFilterHistory;
  uint8_t* temporalFilterPersistenceMap;

  // outputs
  uint16_t* temporalFilterOutput;

  // general parameters
  int width;
  int height;

  // temporal filter params
  bool temporalFilterEnabled;
  uint8_t temporalFilterMask;
  uint16_t temporalFilterDelta;
  float temporalFilterAlpha;
  float temporalFilterOneMinusAlpha;
};

__global__ void temporal_filtering_u16_kernel(TemporalFilterParams bp);