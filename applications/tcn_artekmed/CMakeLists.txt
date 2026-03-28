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
                          tcn_zenoh_publisher
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
                          tcn_flatten_tensor
                          tcn_stream_splitter
                          tcn_stream_merger
                          tcn_shm_serde
                          tcn_shm_subscriber
                          tcn_device_context
                          #slang_shader
                          )

add_holohub_application(tcn_render_tests DEPENDS OPERATORS
        #tcn_slang_renderer
        tcn_depthimage_backprojection
        tcn_depthimage_temporal_filter
        tcn_texture_sampler
        #slang_shader
)

add_holohub_application(tcn_shm_vlm_inference DEPENDS OPERATORS
        #tcn_slang_renderer
        tcn_depthimage_backprojection
        tcn_depthimage_temporal_filter
        tcn_depthimage_weights
        tcn_texture_sampler
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
