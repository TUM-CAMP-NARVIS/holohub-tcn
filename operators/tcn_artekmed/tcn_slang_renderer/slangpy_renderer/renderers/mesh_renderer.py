import slangpy as spy
import numpy as np
from pyglm import glm

from ..renderables.mesh import Mesh

class MeshRenderer:
    def __init__(self, device: spy.Device, output_format: spy.Format):
        self.device = device
        self.program = device.load_program(
            "phong.slang",
            ["vertex_main", "fragment_main"],
            link_options={"debug_info": spy.SlangDebugInfoLevel.maximal},
        )

        self.sampler = device.create_sampler()

        self.pipeline = device.create_render_pipeline(
            program=self.program,
            targets=[{"format": output_format, "enable_blend": False}],
            input_layout=device.create_input_layout(
                input_elements=[
                    {
                        "format": spy.Format.rgb32_float,
                        "semantic_name": "POSITION",
                        "buffer_slot_index": 0,
                    },
                    {
                        "format": spy.Format.rgb32_float,
                        "semantic_name": "NORMAL",
                        "buffer_slot_index": 1,
                    },
                    {
                        "format": spy.Format.rg32_float,
                        "semantic_name": "TEXCOORD",
                        "buffer_slot_index": 2,
                    },
                ],
                vertex_streams=[{"stride": 12}, {"stride": 12}, {"stride": 8}],
            ),
            rasterizer={
                "fill_mode": spy.FillMode.solid,
                "cull_mode": spy.CullMode.none,
                "front_face": spy.FrontFaceMode.counter_clockwise,
                "depth_clip_enable": True,
                "scissor_enable": False,
                "multisample_enable": False,
            },
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
        mesh: Mesh,
        window_size: tuple[int, int],
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        model_matrix: np.ndarray,
        extra_args: dict = None,
    ):
        """
        Render a mesh with the given transformation matrices.
        Note: This method expects to be called within an active render pass.

        Args:
            pass_encoder: Active render pass encoder
            mesh: Mesh object to render
            window_size: (width, height) tuple
            view_matrix: Camera view matrix (4x4)
            proj_matrix: Camera projection matrix (4x4)
            model_matrix: Object pose/model matrix (4x4)
            extra_args: Optional: Additional arguments for rendering customization
        """

        shader_object = pass_encoder.bind_pipeline(self.pipeline)
        cursor = spy.ShaderCursor(shader_object)
        cursor.sampler = self.sampler
        cursor.texture = mesh.texture
        cursor.proj = proj_matrix
        cursor.view = view_matrix
        cursor.inverseView = np.linalg.inv(view_matrix)
        cursor.model = model_matrix

        cursor.renderStaticColor = extra_args.get('renderStaticColor', True)

        if extra_args:
            for k, v in extra_args.items():
                if k not in ['renderStaticColor',]:
                    if cursor.has_field(k):
                        setattr(cursor, k, v)

        if extra_args:
            for k, v in extra_args.items():
                if cursor.has_field(k):
                    setattr(cursor, k, v)

        pass_encoder.set_render_state(
            {
                "viewports": [spy.Viewport.from_size(*window_size)],
                "scissor_rects": [spy.ScissorRect.from_size(*window_size)],
                "vertex_buffers": [
                    mesh.position_buffer,
                    mesh.normal_buffer,
                    mesh.uv_buffer,
                ],
                "index_buffer": mesh.index_buffer,
                "index_format": mesh.index_format,
            }
        )

        pass_encoder.draw_indexed({"vertex_count": mesh.vertex_count})
