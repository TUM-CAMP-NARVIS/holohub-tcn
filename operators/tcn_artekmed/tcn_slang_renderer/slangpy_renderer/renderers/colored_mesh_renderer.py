import slangpy as spy
import numpy as np

from ..renderables.colored_mesh import ColoredMesh


class ColoredMeshRenderer:
    """
    Renderer for reference frame line segments with per-vertex colors.
    """

    def __init__(self, device: spy.Device, output_format: spy.Format):
        self.device = device
        self.program = device.load_program(
            "color_drawable.slang",
            ["vertex_main", "fragment_main"],
            link_options={"debug_info": spy.SlangDebugInfoLevel.maximal},
        )

        # Create render pipeline for line rendering
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
                        "format": spy.Format.rgb32_float,
                        "semantic_name": "COLOR",
                        "buffer_slot_index": 1,
                    },
                ],
                vertex_streams=[{"stride": 12}, {"stride": 12}],
            ),
            depth_stencil={
                "format": spy.Format.d32_float,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
            },
            primitive_topology=spy.PrimitiveTopology.line_list,
            rasterizer={
                "cull_mode": spy.CullMode.none,  # No culling for lines
            }
        )

    def render(
        self,
        pass_encoder: spy.RenderPassEncoder,
        mesh: ColoredMesh,
        window_size: tuple[int, int],
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        model_matrix: np.ndarray,
        extra_args: dict = None,
    ):
        """
        Render a reference frame with colored line segments.
        Note: This method expects to be called within an active render pass.

        Args:
            pass_encoder: Active render pass encoder
            mesh: ReferenceFrame object to render
            window_size: (width, height) tuple
            view_matrix: Camera view matrix (4x4)
            proj_matrix: Camera projection matrix (4x4)
            model_matrix: Object pose/model matrix (4x4)
            extra_args: Additional arguments for rendering
        """

        # Skip rendering if geometry is not ready
        if not mesh.has_geometry:
            return

        shader_object = pass_encoder.bind_pipeline(self.pipeline)
        cursor = spy.ShaderCursor(shader_object)

        # Set transformation matrices
        cursor.proj = proj_matrix
        cursor.view = view_matrix
        cursor.model = model_matrix

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
                    mesh.color_buffer,
                ],
                "index_buffer": mesh.index_buffer,
                "index_format": mesh.index_format,
            }
        )

        # Draw indexed lines
        pass_encoder.draw_indexed({"vertex_count": mesh.index_count})
