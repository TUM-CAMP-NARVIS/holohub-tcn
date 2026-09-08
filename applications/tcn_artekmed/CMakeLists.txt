# SPDX-FileCopyrightText: Copyright (c) 2025 Ulrich Eck
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_holohub_application(tcn_zenoh_receiver DEPENDS OPERATORS
                          tcn_cdr_serde
                          tcn_cdr_decoder
                          tcn_zenoh_subscriber
                          tcn_zenoh_receiver
                          tcn_zenoh_publisher
                          tcn_shm_serde
                          tcn_shm_zenoh_sender
                          nv_video_decoder
                          )

add_holohub_application(tcn_shm_receiver DEPENDS OPERATORS
                          tcn_depthimage_backprojection
                          tcn_depthimage_temporal_filter
                          tcn_depthimage_weights
                          tcn_texture_sampler
                          tcn_convert_bgra_to_rgba
                          tcn_depthimage_apply_mask
                          tcn_depthimage_fgbg_mask
                          tcn_depthimage_max_distance
                          tcn_stream_splitter
                          tcn_stream_merger
                          tcn_shm_serde
                          tcn_shm_subscriber
                          tcn_device_context
                          tcn_flatten_tensor
                          #slang_shader
                          )

add_holohub_application(tcn_shm_backprojection_test DEPENDS OPERATORS
                          tcn_depthimage_backprojection
                          tcn_depthimage_temporal_filter
                          tcn_flatten_tensor
                          tcn_stream_splitter
                          tcn_stream_merger
                          tcn_shm_serde
                          tcn_shm_subscriber
                          tcn_device_context
                          )

add_holohub_application(tcn_render_tests DEPENDS OPERATORS
        #tcn_slang_renderer
        tcn_depthimage_backprojection
        tcn_depthimage_temporal_filter
        tcn_texture_sampler
        tcn_stream_splitter
        tcn_stream_merger
        tcn_shm_serde
        tcn_shm_subscriber
        tcn_device_context
        #slang_shader
)

add_holohub_application(tcn_shm_vlm_inference DEPENDS OPERATORS
        tcn_stream_splitter
        tcn_shm_serde
        tcn_shm_subscriber
        tcn_device_context
        tcn_convert_bgra_to_rgba
        # Fused panoptic-map paint. This DEPENDS entry is what actually enables the build:
        # add_holohub_operator() gates each operator behind an OP_<NAME> cache option that only
        # a DEPENDS OPERATORS listing turns ON. Adding it to the app's target_link_libraries is
        # NOT sufficient -- that is a link-time relationship resolved after configure, so the
        # subdirectory is never added and the module silently is not built.
        tcn_panoptic_map
        # Temporal frame grouping (docs/specs/2026-08-11-temporal-sync-design.md). Was missing:
        # the app imported it and worked only because OP_tcn_stream_synchronizer happened to be ON
        # in an existing build cache, which a fresh configure would not reproduce.
        tcn_stream_synchronizer
        # Mask/depth join (docs/specs/2026-08-12-mask-depth-join-design.md): backprojection
        # supplies the per-depth-pixel texcoords, the label sampler reads the panoptic map through
        # them, and apply_mask consumes the depth-grid mask that falls out.
        tcn_depthimage_backprojection
        tcn_label_sampler
        tcn_labeled_pointcloud
        tcn_instance_stats
        tcn_stream_merger
        tcn_flatten_tensor
        tcn_depthimage_apply_mask
)


add_holohub_application(tcn_shm_ssg_inference DEPENDS OPERATORS
        tcn_depthimage_backprojection
        tcn_depthimage_temporal_filter
        tcn_texture_sampler
        tcn_convert_bgra_to_rgba
        tcn_depthimage_apply_mask
        tcn_depthimage_fgbg_mask
        tcn_depthimage_max_distance
        tcn_stream_splitter
        tcn_stream_merger
        tcn_shm_serde
        tcn_shm_subscriber
        tcn_device_context
        tcn_flatten_tensor
)

add_holohub_application(tcn_operator_tests DEPENDS OPERATORS
        tcn_convert_bgra_to_rgba
        tcn_flatten_tensor
        tcn_depthimage_max_distance
        tcn_depthimage_apply_mask
        tcn_depthimage_fgbg_mask
        tcn_stream_splitter
        tcn_stream_merger
)



add_holohub_application(tcn_all DEPENDS OPERATORS
        tcn_cdr_decoder
        tcn_cdr_serde
        tcn_convert_bgra_to_rgba
        tcn_depthimage_apply_mask
        tcn_depthimage_backprojection
        tcn_depthimage_fgbg_mask
        tcn_depthimage_max_distance
        tcn_depthimage_temporal_filter
        tcn_depthimage_weights
        tcn_device_context
        tcn_flatten_tensor
        tcn_instance_stats
        tcn_label_sampler
        tcn_labeled_pointcloud
        tcn_panoptic_map
        tcn_shm_serde
        tcn_shm_subscriber
        tcn_shm_zenoh_sender
        tcn_stream_merger
        tcn_stream_splitter
        tcn_stream_synchronizer
        tcn_texture_sampler
        tcn_zenoh_publisher
        tcn_zenoh_receiver
        tcn_zenoh_subscriber
)
