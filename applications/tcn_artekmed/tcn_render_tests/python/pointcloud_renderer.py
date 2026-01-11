import slangpy as spy
import numpy as np
import logging

from pointcloud_data import Pointcloud

log = logging.getLogger(__name__)

class PointcloudRenderer:
    def __init__(self, device: spy.Device, output_format: spy.Format):
        self.device = device
        self.program = device.load_program(
            "pointcloud.slang",
            ["vertex_main", "fragment_main"],
            link_options={"debug_info": spy.SlangDebugInfoLevel.maximal},
        )

        self.sampler = device.create_sampler()

        self.pipeline = device.create_render_pipeline(
            program=self.program,
            targets=[{"format": output_format}],
            input_layout=device.create_input_layout(
                input_elements=[
                    {
                        "format": spy.Format.rgb32_float,
                        "semantic_name": "POSITION",
                        "buffer_slot_index": 0,
                    },
                    {
                        "format": spy.Format.rg32_float,
                        "semantic_name": "TEXCOORD",
                        "buffer_slot_index": 1,
                    },
                ],
                vertex_streams=[{"stride": 12}, {"stride": 8}],
            ),
            primitive_topology=spy.PrimitiveTopology.point_list,
            depth_stencil={
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
            },
        )

    def render(
        self,
        command_encoder: spy.CommandEncoder,
        pointcloud: Pointcloud,
        window_size: tuple[int, int],
        output_texture: spy.Texture,
        depth_texture: spy.Texture,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
               model_matrix: np.ndarray,
        clear_color: list = None,
        extra_args: dict = None,
    ):
        """
        Render a pointcloud with the given transformation matrices.

        Args:
            command_encoder: Slang command encoder
            pointcloud: Pointcloud object to render
            window_size: (width, height) tuple
            output_texture: Target texture
            depth_texture: Depth buffer
            view_matrix: Camera view matrix (4x4)
            proj_matrix: Camera projection matrix (4x4)
            model_matrix: Object pose/model matrix (4x4)
            clear_color: RGBA clear color, or None to skip clearing
            extra_args: Optional: Additional arguments for rendering customization
        """

        # Skip rendering if essential data is missing
        if not (pointcloud.has_vertices and pointcloud.has_texcoords and pointcloud.has_texture):
            log.debug(f"Pointcloud is incomplete..")
            return

        with command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": output_texture.create_view(),
                        "clear_value": clear_color if clear_color else [0.2, 0.2, 0.2, 1.0],
                        "load_op": spy.LoadOp.clear if clear_color else spy.LoadOp.load,
                    }
                ],
                "depth_stencil_attachment": {
                    "view": depth_texture.create_view(),
                    "depth_load_op": spy.LoadOp.clear if clear_color else spy.LoadOp.load,
                },
            }
        ) as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.sampler = self.sampler
            cursor.texture = pointcloud.texture
            cursor.proj = proj_matrix
            cursor.view = view_matrix
            cursor.model = model_matrix
            for k, v in extra_args.items():
                if hasattr(cursor, k):
                    setattr(cursor, k, v)

            pass_encoder.set_render_state(
                {
                    "viewports": [spy.Viewport.from_size(*window_size)],
                    "scissor_rects": [spy.ScissorRect.from_size(*window_size)],
                    "vertex_buffers": [
                        pointcloud.position_buffer,
                        pointcloud.uv_buffer,
                    ],
                }
            )
            pass_encoder.draw({"vertex_count": pointcloud.vertices.size})
