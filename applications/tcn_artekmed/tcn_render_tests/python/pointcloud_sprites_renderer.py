import slangpy as spy
import numpy as np
import logging

from pointcloud_data import Pointcloud

log = logging.getLogger(__name__)

class PointcloudSpritesRenderer:
    def __init__(self, device: spy.Device, output_format: spy.Format):
        self.device = device
        self.program = device.load_program(
            "pointcloud_sprites.slang",
            ["vertex_main", "geometry_main", "fragment_main"],
            link_options={"debug_info": spy.SlangDebugInfoLevel.maximal},
        )

        self.sampler = device.create_sampler()

        # Note: This shader uses a geometry shader, so no input layout is needed
        # The vertex shader uses SV_VertexID directly
        self.pipeline = device.create_render_pipeline(
            program=self.program,
            input_layout=None,
            targets=[{"format": output_format}],
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
        Render a pointcloud with sprites using the geometry shader.

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
                       Expected parameters based on pointcloud_sprites.slang:
                       - transform: float4x4 (additional transformation matrix)
                       - pointCloud: StructuredBuffer<Point3D> (point cloud data buffer)
                       - colorTex: Texture2D<float4> (RGB color texture)
                       - bodyIndexTex: Texture2D<uint> (body segmentation texture)
                       - staticColor: float4 (fallback color)
                       - pointSize: float (sprite size)
                       - enableClipping: float (enable clipping flag)
                       - clipScale: float (scale factor for clipping)
                       - worldToBox: float4x4 (world to clipping box transform)
                       - roI2Mag: float4x4 (region of interest to magnification transform)
                       - scaleCenter: float4 (center point for scaling)
                       - boxPosition_1/2/3: float4 (clipping box positions)
                       - boxSize_1/2/3: float4 (clipping box sizes)
                       - depthWidth: int (depth image width)
                       - depthHeight: int (depth image height)
                       - depthMultiple: int (TODO: clarify usage)
                       - processingRadius: uint (TODO: clarify usage)
                       - useStaticColor: bool (use staticColor instead of texture)
                       - drawUnconnected: bool (sprite mode vs mesh mode)
                       - hasBodyIndex: bool (enable body index filtering)
                       - dontRenderPlayers: bool (skip player points)
                       - dontRenderEnvironment: bool (skip environment points)
                       - activateAdditionalBox: bool (enable third clipping box)
                       - useClippingBox: bool (enable clipping boxes 1 and 2)
                       - screenParams: float2 (screen dimensions for aspect ratio)
        """

        # Skip rendering if essential data is missing
        # TODO: Determine minimal requirements - shader uses structured buffer, not vertex buffers
        if not pointcloud.has_vertices:
            log.debug(f"Pointcloud is incomplete (no vertices)..")
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
                    "depth_clear_value": 1.0,
                    "depth_load_op": spy.LoadOp.clear if clear_color else spy.LoadOp.load,
                    "depth_store_op": spy.StoreOp.store,
                },
            }
        ) as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)

            # Set core transformation matrices
            cursor.proj = proj_matrix
            cursor.view = view_matrix
            cursor.model = model_matrix

            # TODO: Determine how to set 'transform' matrix - shader expects this separately from model/view/proj
            cursor.transform = np.eye(4, dtype=np.float32)

            # Bind separate buffers for vertices and UV coordinates
            # The shader expects:
            # - uniform StructuredBuffer<float3> vertices;
            # - uniform StructuredBuffer<float2> uvCoords;
            if pointcloud.has_vertices:
                cursor.vertices = pointcloud.position_buffer  # float3 (rgb32_float)
            if pointcloud.has_texcoords:
                cursor.uvCoords = pointcloud.uv_buffer  # float2 (rg32_float)

            # Set texture and sampler if available
            if pointcloud.has_texture:
                cursor.colorTex = pointcloud.texture
                cursor.sampler_colorTex = self.sampler

            # TODO: Determine source of bodyIndexTex (body segmentation data)
            # cursor.bodyIndexTex = ???

            # Set screen parameters for sprite aspect ratio correction
            cursor.screenParams = np.array([window_size[0], window_size[1]], dtype=np.float32)

            # Initialize extra_args if not provided
            if extra_args is None:
                extra_args = {}

            # Set shader uniforms with defaults from extra_args
            cursor.staticColor = extra_args.get('staticColor', np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
            cursor.pointSize = extra_args.get('pointSize', 0.008)  # Default sprite size in clip space
            cursor.useStaticColor = extra_args.get('useStaticColor', False)
            cursor.drawUnconnected = extra_args.get('drawUnconnected', True)  # True for sprites, False for mesh
            cursor.hasBodyIndex = extra_args.get('hasBodyIndex', False)
            cursor.dontRenderPlayers = extra_args.get('dontRenderPlayers', False)
            cursor.dontRenderEnvironment = extra_args.get('dontRenderEnvironment', False)
            cursor.enableClipping = extra_args.get('enableClipping', 0.0)
            cursor.useClippingBox = extra_args.get('useClippingBox', False)
            cursor.activateAdditionalBox = extra_args.get('activateAdditionalBox', False)

            # TODO: Get from pointcloud source
            cursor.depthWidth = extra_args.get('depthWidth', 0)
            cursor.depthHeight = extra_args.get('depthHeight', 0)

            # Note: This shader doesn't use vertex buffers in the traditional sense
            # The geometry shader reads from a StructuredBuffer instead
            # The vertex count determines how many points are processed
            pass_encoder.set_render_state(
                {
                    "viewports": [spy.Viewport.from_size(*window_size)],
                    "scissor_rects": [spy.ScissorRect.from_size(*window_size)],
                }
            )

            # Draw call: vertex_count should match the number of points in the point cloud
            # TODO: Determine the correct vertex count from pointcloud data structure
            pass_encoder.draw({"vertex_count": pointcloud.vertices.size if hasattr(pointcloud.vertices, 'size') else len(pointcloud.vertices)})
