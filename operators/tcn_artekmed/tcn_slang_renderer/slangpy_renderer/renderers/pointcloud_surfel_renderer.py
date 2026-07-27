import slangpy as spy
import numpy as np
import logging

from ..renderables.pointcloud import Pointcloud
from ..utils.depth_unprojector import (
    ColorProjectionParameters,
    _bind_intrinsics,
    _bind_color_projection_params,
)

log = logging.getLogger(__name__)


class PointcloudSurfelRenderer:
    """
    Renderer for normal-oriented textured hexagonal surfels.

    Uses a geometry shader to expand each point into a hexagonal surfel
    oriented by the point's normal vector. Each surfel vertex gets a UV
    by projecting its 3D position through the color camera model.
    """

    def __init__(self, device: spy.Device, output_format: spy.Format):
        self.device = device
        self.program = device.load_program(
            "pointcloud_surfels.slang",
            ["vertex_main", "geometry_main", "fragment_main"],
        )

        self.sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.point,
            mag_filter=spy.TextureFilteringMode.point,
            mip_filter=spy.TextureFilteringMode.point,
        )

        self.pipeline = device.create_render_pipeline(
            program=self.program,
            input_layout=None,
            targets=[{
                "format": output_format,
                "enable_blend": True,
                "color": {
                    "src_factor": spy.BlendFactor.src_alpha,
                    "dst_factor": spy.BlendFactor.inv_src_alpha,
                    "op": spy.BlendOp.add,
                },
                "alpha": {
                    "src_factor": spy.BlendFactor.one,
                    "dst_factor": spy.BlendFactor.inv_src_alpha,
                    "op": spy.BlendOp.add,
                },
            }],
            primitive_topology=spy.PrimitiveTopology.triangle_strip,
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
        Render a pointcloud as normal-oriented textured hexagonal surfels.

        Args:
            pass_encoder: Active render pass encoder
            pointcloud: Pointcloud with position_buffer, normal_buffer, and texture
            window_size: (width, height) tuple
            view_matrix: Camera view matrix (4x4)
            proj_matrix: Camera projection matrix (4x4)
            model_matrix: Object pose/model matrix (4x4)
            extra_args: Optional dict with keys:
                - color_params: ColorProjectionParameters (for per-vertex UV projection)
                - depth_fy: float (depth camera fy for auto-sizing)
                - sprite_scale: float (default 1.5, controls surfel overlap)
                - useStaticColor: bool (debug: render with flat color)
                - staticColor: np.ndarray (float4, default white)
                - depthWidth: int
                - depthHeight: int
        """
        if not pointcloud.has_vertices:
            log.debug("Pointcloud has no vertices, skipping surfel render")
            return

        if not pointcloud.has_normals:
            log.debug("Pointcloud has no normals, skipping surfel render")
            return

        extra_args = extra_args or {}

        shader_object = pass_encoder.bind_pipeline(self.pipeline)
        cursor = spy.ShaderCursor(shader_object)

        # Transformation matrices
        cursor.proj = proj_matrix
        cursor.view = view_matrix
        cursor.model = model_matrix

        # Structured buffers
        cursor.vertices = pointcloud.position_buffer
        cursor.normals = pointcloud.normal_buffer
        if pointcloud.has_texcoords:
            cursor.uvCoords = pointcloud.uv_buffer

        # Texture + sampler
        if pointcloud.has_texture:
            cursor.colorTex = pointcloud.texture
            cursor.sampler_colorTex = self.sampler

        # Color camera parameters for per-vertex UV projection
        color_params = extra_args.get("color_params")
        if color_params is not None:
            cursor.has_color_projection = True
            _bind_color_projection_params(cursor.color_camera, color_params)
        else:
            cursor.has_color_projection = False
            _bind_color_projection_params(cursor.color_camera, None)

        # Sprite sizing
        cursor.depth_fy = extra_args.get("depth_fy", 500.0)
        cursor.sprite_scale = extra_args.get("sprite_scale", 1.5)

        # Edge falloff
        cursor.falloff_exponent = extra_args.get("falloff_exponent", 4.0)

        # Depth dimensions for structured grid
        cursor.depthWidth = extra_args.get("depthWidth", 0)
        cursor.depthHeight = extra_args.get("depthHeight", 0)

        # Debug options
        cursor.useStaticColor = extra_args.get("useStaticColor", False)
        cursor.staticColor = extra_args.get(
            "staticColor", np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        )

        # Render state
        pass_encoder.set_render_state(
            {
                "viewports": [spy.Viewport.from_size(*window_size)],
                "scissor_rects": [spy.ScissorRect.from_size(*window_size)],
            }
        )

        # Draw: one vertex per point, geometry shader expands to hexagon
        vertex_count = (
            pointcloud.vertices.size
            if hasattr(pointcloud.vertices, "size")
            else len(pointcloud.vertices)
        )
        pass_encoder.draw({"vertex_count": vertex_count})
