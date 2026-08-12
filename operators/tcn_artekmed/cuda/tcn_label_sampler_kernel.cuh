/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

#include "../common/datatypes.hpp"

/// Number of distinct class ids representable by the packed panoptic encoding. The encoding is
/// `(class_id << 8) | instance_id` in a uint16 (see `langsam_helpers.py`, which does the packing),
/// so a class id occupies the high byte and there are exactly 256 of them. This is the size of the
/// class-selection lookup table, which is why it is a compile-time constant rather than a parameter.
constexpr int kPanopticNumClasses = 256;

/// Shift that separates class from instance in a packed panoptic label. MUST match the packing in
/// `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py`; the two are the only
/// places the convention is encoded, and a mismatch silently reassigns every label to a wrong class.
constexpr int kPanopticClassShift = 8;

struct LabelSamplerParams {
  const uint16_t* labels;       ///< [labelHeight*labelWidth] packed panoptic map (colour grid)
  const float2*   uv;           ///< [height*width] normalised texcoords, NaN = no correspondence

  // outputs
  uint16_t* labelsOut;          ///< [height*width] packed label per depth pixel
  uint8_t*  maskOut;            ///< [height*width] 255 where a selected class was hit, else 0

  /// [kPanopticNumClasses] 0/1 per class id, or nullptr to select every non-background class.
  const uint8_t* classSelect;

  // general parameters
  int width;                    ///< depth-grid width  (== texcoord width)
  int height;                   ///< depth-grid height (== texcoord height)
  int labelWidth;               ///< panoptic-map width
  int labelHeight;              ///< panoptic-map height
  uint16_t unlabeled;           ///< written where no label can be assigned
};

__global__ void label_sampler_nearest_kernel(LabelSamplerParams bp);
