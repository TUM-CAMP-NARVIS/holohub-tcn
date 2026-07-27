import slangpy as spy
import numpy as np
import logging

from ..renderables.pointcloud import Pointcloud

log = logging.getLogger(__name__)

class PointcloudRenderer:
    def __init__(self, device: spy.Device, output_format: spy.Format):
        self.device = device
        self.program = device.load_program(
            "pointcloud.slang",
            ["vertex_main", "fragment_main"],
            link_options={"debug_info": spy.SlangDebugInfoLevel.maximal, 'optimization':spy.SlangOptimizationLevel.none},
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
                "format": spy.Format.d32_float,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
            },
        )

    def render(
        self,
        pass_encoder: spy.RenderPassEncoder,
        pointcloud: Pointcloud,
        window_size: tuple[int, int],
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        model_matrix: np.ndarray,
        extra_args: dict = None,
    ):
        """
        Render a pointcloud with the given transformation matrices.
        Note: This method expects to be called within an active render pass.

        Args:
            pass_encoder: Active render pass encoder
            pointcloud: Pointcloud object to render
            window_size: (width, height) tuple
            view_matrix: Camera view matrix (4x4)
            proj_matrix: Camera projection matrix (4x4)
            model_matrix: Object pose/model matrix (4x4)
            extra_args: Optional: Additional arguments for rendering customization
        """

        # Skip rendering if essential data is missing
        if not (pointcloud.has_vertices and pointcloud.has_texcoords and pointcloud.has_texture):
            log.debug(f"Pointcloud is incomplete..")
            return

        shader_object = pass_encoder.bind_pipeline(self.pipeline)
        cursor = spy.ShaderCursor(shader_object)
        cursor.sampler = self.sampler
        cursor.texture = pointcloud.texture
        cursor.proj = proj_matrix
        cursor.view = view_matrix
        cursor.model = model_matrix

        # uniforms
        cursor.renderStaticColor = extra_args.get('renderStaticColor', False)
        cursor.pointSize = extra_args.get('pointSize', 1.0)

        if extra_args:
            for k, v in extra_args.items():
                if k not in ['renderStaticColor', 'pointSize']:
                    if cursor.has_field(k):
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
