/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Nearest-neighbour sampling of a packed panoptic label map through per-depth-pixel texcoords.
 *
 * Deliberately NOT a variant of tcn_texture_sampler_kernel.cu: that kernel interpolates bilinearly,
 * which is right for colour and meaningless for labels. Averaging four packed
 * `(class << 8) | instance` ids yields an id denoting none of the classes involved, so every object
 * boundary would be assigned an invented label. Label images admit exactly one filter.
 */
#include <cuda_runtime.h>
#include <cmath>
#include "tcn_label_sampler_kernel.cuh"

__global__ void label_sampler_nearest_kernel(LabelSamplerParams bp) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= bp.width || y >= bp.height) return;
  const int idx = y * bp.width + x;

  const float2 uv = bp.uv[idx];

  // Two distinct rejections, both meaning "this depth pixel has no label", kept separate from a
  // successful sample so that a configured non-zero `unlabeled` value can never be mistaken for a
  // real class by the mask below:
  //   - non-finite: backprojection writes NaN when the depth sample is outside near/far or invalid
  //   - outside [0,1]: the 3D point projects outside the colour frustum. NOT clamped, unlike the
  //     colour sampler: edge extension is reasonable for colour and wrong for labels, where it
  //     would smear the border object across everything beside it.
  const bool valid = isfinite(uv.x) && isfinite(uv.y) &&
                     uv.x >= 0.0f && uv.x <= 1.0f && uv.y >= 0.0f && uv.y <= 1.0f;

  uint16_t label = bp.unlabeled;
  if (valid) {
    // Same [0,1] -> [0, N-1] mapping as the colour sampler, so colour and label sampling address
    // one pixel grid; rounded instead of floored because the nearest sample is the correct one.
    int lx = static_cast<int>(rintf(uv.x * static_cast<float>(bp.labelWidth  - 1)));
    int ly = static_cast<int>(rintf(uv.y * static_cast<float>(bp.labelHeight - 1)));
    lx = max(0, min(lx, bp.labelWidth  - 1));   // rounding cannot exceed the range; cheap and final
    ly = max(0, min(ly, bp.labelHeight - 1));
    label = bp.labels[ly * bp.labelWidth + lx];
  }

  bp.labelsOut[idx] = label;

  if (bp.maskOut) {
    uint8_t selected = 0;
    if (valid && label != 0) {                  // 0 is background in the packed encoding
      const int cls = label >> kPanopticClassShift;
      selected = (bp.classSelect == nullptr || bp.classSelect[cls] != 0) ? 255 : 0;
    }
    bp.maskOut[idx] = selected;
  }
}
