#
# Place the license header here
#
import logging
import os
from argparse import ArgumentParser
import threading
import json

from typing import Callable, Optional, Union
import slangpy as spy
from pathlib import Path

import iceoryx2 as iox2

from pyglm import glm

import numpy as np
import cupy as cp
import holoscan as hs
import torch

from holoscan.core import Operator, OperatorSpec, Tracker, ConditionType, IOSpec
from holoscan.conditions import CountCondition, PeriodicCondition, BooleanCondition
from holoscan.logger import LogLevel, set_log_level
from holoscan.operators import HolovizOp, holoviz
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler
from holoscan.resources import CudaStreamPool, BlockMemoryPool, MemoryStorageType, RMMAllocator
from holoscan.pose_tree import PoseTreeManager, SO3, Pose3

from holohub.tcn_depthimage_backprojection import TcnDepthImageBackprojectionOp
from operators.tcn_artekmed.tcn_processing import ShmSimpleBackprojectionSubgraph

from operators.tcn_artekmed.tcn_util.helpers import pose3_to_matrix4x4

from pointcloud_renderer import PointcloudRenderer, Pointcloud
from pointcloud_sprites_renderer import PointcloudSpritesRenderer
from colored_mesh_renderer import ColoredMeshRenderer, ColoredMesh
from mesh_renderer import MeshRenderer, Mesh

from renderable import Renderable
from arcball_controller import ArcBall
from fpv_controller import FirstPersonView
#from graph_visualizer import visualize_holoscan_graph
import time
from dataclasses import dataclass


log = logging.getLogger(__name__)

CORRECTION_VK = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# testing different method to compute projection matrix:
@dataclass
class AsymmetricPerspectiveOptions:
    left: float = -1.0
    right: float = 1.0
    bottom: float = -1.0
    top: float = 1.0
    nearPlane: float = 0.1
    farPlane: float = 100.0
    applyPostViewCorrection: bool = True


def get_frustum_extents_from_fov(fov_deg: float, aspect: float, near: float):
    """
    Computes left, right, bottom, top extents at the near plane
    based on vertical FOV and aspect ratio.
    """
    top = near * np.tan(np.radians(fov_deg) / 2.0)
    bottom = -top
    right = top * aspect
    left = -right
    return left, right, bottom, top


def perspective_asymmetric(options: AsymmetricPerspectiveOptions) -> glm.mat4:
    """
    Translated from C++. Computes a Vulkan-compatible projection matrix.
    Note: pyglm matrix constructor is column-major.
    """
    two_near = 2.0 * options.nearPlane
    right_minus_left = options.right - options.left
    far_minus_near = options.farPlane - options.nearPlane

    if not options.applyPostViewCorrection:
        bottom_minus_top = options.bottom - options.top

        # Column-major construction
        m = glm.mat4(
            two_near / right_minus_left, 0.0, 0.0, 0.0,  # Col 0
            0.0, two_near / bottom_minus_top, 0.0, 0.0,  # Col 1
            -(options.right + options.left) / right_minus_left,  # Col 2
            -(options.bottom + options.top) / bottom_minus_top,
            options.farPlane / far_minus_near,
            1.0,
            0.0, 0.0, -options.nearPlane * options.farPlane / far_minus_near, 0.0  # Col 3
        )
        return m
    else:
        # Flip signs for top/bottom to account for 180 deg X-axis rotation
        bottom = -options.bottom
        top = -options.top
        bottom_minus_top = bottom - top

        # Column-major construction with negated Y and Z axes
        m = glm.mat4(
            two_near / right_minus_left, 0.0, 0.0, 0.0,  # Col 0
            0.0, -two_near / bottom_minus_top, 0.0, 0.0,  # Col 1
            (options.right + options.left) / right_minus_left,  # Col 2
            (bottom + top) / bottom_minus_top,
            -options.farPlane / far_minus_near,
            -1.0,
            0.0, 0.0, -options.nearPlane * options.farPlane / far_minus_near, 0.0  # Col 3
        )
        return m





class SlangWindow:
    def __init__(self, width: int, height: int, title: str, resizeable: bool = True, close_callback: Callable = None):
        self.ui = None
        self.close_callback = close_callback

        self.window = spy.Window(width, height, title, resizable=resizeable)
        asset_root_dir = Path(__file__).parent / "assets"

        with cp.cuda.Device(0):
            _ = cp.zeros((1,), dtype=cp.uint8)  # forces context creation if not existant
            device_handle = spy.get_cuda_current_context_native_handles()

        self.device = spy.Device(
            enable_debug_layers=True,
            enable_cuda_interop=True,
            existing_device_handles=device_handle,
            compiler_options={"include_paths": [
                str(asset_root_dir / "shaders"),
                os.path.join(os.path.dirname(spy.__file__), "slang"),
            ]},
        )

        self.surface = self.device.create_surface(self.window)
        self.surface.configure(self.window.width, self.window.height)

        # Create renderers (stateless, shared by all renderables)
        self.mesh_renderer = MeshRenderer(self.device, self.surface.config.format)
        self.pointcloud_renderer = PointcloudRenderer(self.device, self.surface.config.format)
        self.pointcloud_sprites_renderer = PointcloudSpritesRenderer(self.device, self.surface.config.format)
        self.colored_mesh_renderer = ColoredMeshRenderer(self.device, self.surface.config.format)

        # Scene management
        self._renderables = {}  # name -> Renderable
        self._next_id = 0

        # Camera setup
        self.camera_pos = np.asarray([5, 5, 5], dtype=np.float32)
        self.camera_target = np.asarray([0, 0, 0], dtype=np.float32)
        self.camera_up = np.asarray([0, 1, 0], dtype=np.float32)
        self.fov = 60.0

        self.model_pose = spy.math.float3(0., 0., 0.)

        self.near_plane = 0.1
        self.far_plane = 10.0
        self.timer = time.perf_counter()

        # self.arc_ball = ArcBall(self.camera_pos, self.camera_target, self.camera_up, self.fov, (width, height))
        self.arc_ball = FirstPersonView(self.camera_pos, self.camera_target, self.camera_up, self.fov, (width, height))
        self.current_mouse_button_down = None
        self.arc_ball_needs_init = False

        self.window.on_keyboard_event = self._on_window_keyboard_event
        self.window.on_mouse_event = self._on_window_mouse_event
        self.window.on_resize = self.handle_resize

        # ui variables
        self._render_static_colors = False


        # create ui
        self.create_depth_texture()
        self.setup_ui()
        self.dirty = False
        self.surface_texture = None
        self._mouse_pos = None

        # sync rendering with data input
        self._cv = threading.Condition()
        self._should_render = True
        self._running = True


        # Hookable events
        self.on_keyboard_event: Optional[Callable[[spy.KeyboardEvent], None]] = None
        self.on_mouse_event: Optional[Callable[[spy.MouseEvent], None]] = None

        # Load a default mesh for testing (can be removed later)
        # model_path = asset_root_dir / "models" / "monkey.obj"
        # default_mesh = Mesh.from_obj(self.device, str(model_path))
        # # default_mesh.pose = Pose3.from_translation(np.asarray([0, 0, 0.5], dtype=np.float32))
        # self.add_renderable("default_mesh", default_mesh)

    # helper
    def set_model_pose(self, pose: spy.math.float3):
        self.model_pose = pose
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = pose[0]
        transform[1, 3] = pose[1]
        transform[2, 3] = pose[2]
        for renderable in self._renderables.values():
            renderable.pose = transform

    def setup_ui(self):
        self.ui = spy.ui.Context(self.device)

        window = spy.ui.Window(
            self.ui.screen, "Settings", spy.float2(10, 10), spy.float2(300, 300)
        )

        spy.ui.CheckBox(window, "Render Static Color", self._render_static_colors, lambda v: setattr(self, "_render_static_colors", v))
        spy.ui.InputFloat3(window, "Model Pose", self.model_pose, self.set_model_pose)


    def add_renderable(self, name: str, renderable: Renderable, pose: np.ndarray = None) -> str:
        """
        Add a renderable object to the scene.

        Args:
            name: Unique name for this renderable
            renderable: Renderable object (Mesh, Pointcloud, etc.)
            pose: Optional 4x4 transformation matrix (defaults to identity)

        Returns:
            The name used to identify this renderable

        Raises:
            ValueError: If name already exists
        """
        if name in self._renderables:
            raise ValueError(f"Renderable with name '{name}' already exists")

        # Associate renderer with the renderable
        # @todo: make renderers pluggable via registry and/or function argument
        if isinstance(renderable, Mesh):
            renderable.renderer = self.mesh_renderer
        elif isinstance(renderable, Pointcloud):
            renderable.renderer = self.pointcloud_renderer
            # renderable.renderer = self.pointcloud_sprites_renderer
        elif isinstance(renderable, ColoredMesh):
            renderable.renderer = self.colored_mesh_renderer

        if pose is not None:
            renderable.pose = pose

        self._renderables[name] = renderable
        return name

    def remove_renderable(self, name: str):
        """Remove a renderable object from the scene."""
        if name in self._renderables:
            del self._renderables[name]

    def get_renderable(self, name: str) -> Optional[Renderable]:
        """Get a renderable by name."""
        return self._renderables.get(name)

    def set_pose(self, name: str, pose: np.ndarray):
        """Set the 6D pose of a renderable object."""
        if name in self._renderables:
            self._renderables[name].pose = pose

    def set_visible(self, name: str, visible: bool):
        """Set the visibility of a renderable object."""
        if name in self._renderables:
            self._renderables[name].visible = visible

    def get_view_matrix(self) -> np.ndarray:
        """Compute the current view matrix from camera parameters."""
        return self.arc_ball.view_matrix()
        # return glm.lookAt(self.camera_pos.tolist(), self.camera_target.tolist(), self.camera_up.tolist())

    def get_projection_matrix(self) -> np.ndarray:
        """Compute the current projection matrix from camera parameters."""
        # aspect = float(self.window.width) / float(self.window.height)
        # return glm.perspective(glm.radians(self.fov), aspect, self.near_plane, self.far_plane)

        aspect = float(self.window.width) / float(self.window.height)
        proj_matrix = glm.perspectiveRH_ZO(glm.radians(self.fov), aspect, self.near_plane, self.far_plane)
        return proj_matrix # @ CORRECTION_VK

        # aspect = float(self.window.width) / float(self.window.height)
        # return self.perspective_vulkan(self.fov, aspect, self.near_plane, self.far_plane)

        # aspect = float(self.window.width) / float(self.window.height)
        # l, r, b, t = get_frustum_extents_from_fov(self.fov, aspect, self.near_plane)
        # opts = AsymmetricPerspectiveOptions(left=l, right=r, bottom=b, top=t, nearPlane=self.near_plane, farPlane=self.far_plane)
        # opts.applyPostViewCorrection = False
        # proj_matrix = perspective_asymmetric(opts)
        # return proj_matrix

    @staticmethod
    def perspective_vulkan(fov_deg, aspect, near, far):
        """
        Computes a right-handed perspective projection matrix for Vulkan/Slang.
        Depth range is [0, 1].
        """
        f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)

        # Note the standard Vulkan/D3D projection:
        # m[0,0] = f / aspect
        # m[1,1] = -f (to flip Y if screen space is top-down, but slangpy handles viewport usually)
        # However, to match standard expected orientation:
        res = np.zeros((4, 4), dtype=np.float32)
        res[0, 0] = f / aspect
        res[1, 1] = f
        res[2, 2] = far / (near - far)
        res[2, 3] = (near * far) / (near - far)
        res[3, 2] = -1.0

        return res

    @staticmethod
    def look_at(eye, target, up):
        """Native numpy implementation of lookAt (Right-Handed)"""
        zaxis = eye - target
        zaxis /= np.linalg.norm(zaxis)
        xaxis = np.cross(up, zaxis)
        xaxis /= np.linalg.norm(xaxis)
        yaxis = np.cross(zaxis, xaxis)

        res = np.eye(4, dtype=np.float32)
        res[0, 0:3] = xaxis
        res[1, 0:3] = yaxis
        res[2, 0:3] = zaxis
        res[0, 3] = -np.dot(xaxis, eye)
        res[1, 3] = -np.dot(yaxis, eye)
        res[2, 3] = -np.dot(zaxis, eye)
        return res


    def get_device(self):
        return self.device

    def close(self):
        self._running = False
        self.window.close()
        if self.close_callback is not None:
            self.close_callback()

    def handle_resize(self, width, height):
        self.dirty = True

    def create_depth_texture(self):
        self.depth_texture = self.device.create_texture(
            format=spy.Format.d32_float,
            width=self.window.width,
            height=self.window.height,
            usage=spy.TextureUsage.depth_stencil,
        )

    def resize(self):
        del self.depth_texture
        del self.surface_texture
        self.device.wait()
        self.surface.configure(self.window.width, self.window.height)
        self.create_depth_texture()

    def _on_visibility_changed(self, name: str, value: bool):
        if name in self._renderables:
            self._renderables[name].visible = value

    def _on_window_keyboard_event(self, event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.close()
                return

            key_str = chr(event.key.value)
            if key_str in [str(i+1) for i in range(9)]:
                idx = int(key_str) - 1
                keys = list(sorted(self._renderables.keys()))
                if idx < len(keys):
                    self.set_visible(keys[idx], not self._renderables[keys[idx]].visible)
        if self.on_keyboard_event:
            self.on_keyboard_event(event)
        else:
            self.ui.handle_keyboard_event(event)

    def _on_window_mouse_event(self, event: spy.MouseEvent):
        if event.type == spy.MouseEventType.button_down:
            log.debug(f"Mouse button down {event.pos} {event.mods} {event.button}")
            self.current_mouse_button_down = event.button
            self.arc_ball_needs_init = True
        elif event.type == spy.MouseEventType.button_up:
            log.debug(f"Mouse button up {event.pos} {event.mods} {event.button}")
            self.current_mouse_button_down = None
        elif event.type == spy.MouseEventType.move:
            pos = (int(event.pos.x), int(event.pos.y))
            if self.current_mouse_button_down == spy.MouseButton.left:
                if self.arc_ball_needs_init:
                    self.arc_ball.init_transformation(pos)
                    self.arc_ball_needs_init = False

                if event.mods == spy.KeyModifierFlags.shift:
                    self.arc_ball.translate(pos)
                else:
                    self.arc_ball.rotate(pos)
                self.request_redraw()
        elif event.type == spy.MouseEventType.scroll:
            log.debug(f"mouse scroll {event}")
            delta = event.scroll.y / 5.0
            if event.mods == spy.KeyModifierFlags.shift:
                delta /= 3.0
            self.arc_ball.zoom(delta)
            self.request_redraw()

        self.ui.handle_mouse_event(event)

    # def draw_ui_controls(self):
    #     self.ui_settings_camerapose =
    #     for k,v in self._renderables.items()
    #         spy.ui.CheckBox()


    def request_redraw(self):
        with self._cv:
            self._should_render = True
            self._cv.notify()

    def run(self):
        while self._running:
            self.window.process_events()

            if self.window.should_close():
                self._running = False
                break

            with self._cv:
                if not self._should_render:
                    self._cv.wait(timeout=0.01)
                should_render = self._should_render
                self._should_render = False

            if not should_render:
                continue

            # now render the frame (maybe in a separate method

            window_size = (self.window.width, self.window.height)
            if self.dirty:
                self.resize()
                self.arc_ball.reshape(window_size)
                # XX arcball resize missing
                self.dirty = False

            self.arc_ball.update_transformation()

            self.surface_texture = self.surface.acquire_next_image()
            if not self.surface_texture:
                continue

            command_encoder = self.device.create_command_encoder()
            # self.ui.begin_frame(*window_size)

            # Compute camera matrices once per frame
            view_matrix = self.get_view_matrix()
            proj_matrix = self.get_projection_matrix()

            # Sync GPU buffers for all dirty renderables before rendering
            for name, renderable in self._renderables.items():
                if renderable.visible:
                    renderable.sync_gpu()

            # Begin single render pass for all renderables
            with command_encoder.begin_render_pass(
                {
                    "color_attachments": [
                        {
                            "view": self.surface_texture.create_view(),
                            "clear_value": [0.0, 0.0, 0.0, 1.0],
                            "load_op": spy.LoadOp.clear,
                        }
                    ],
                    "depth_stencil_attachment": {
                        "view": self.depth_texture.create_view(),
                        # "depth_clear_value": 1.0,
                        "depth_load_op": spy.LoadOp.clear,
                        "depth_store_op": spy.StoreOp.store,
                    },
                }
            ) as pass_encoder:
                # Render all visible renderables in a single pass
                for name, renderable in self._renderables.items():
                    if not renderable.visible:
                        continue

                    # Use the render method from the renderable (which delegates to its renderer)
                    renderable.render(
                        pass_encoder,
                        window_size,
                        view_matrix,
                        proj_matrix,
                        extra_args={"renderStaticColor": self._render_static_colors}
                    )

            # self.ui.end_frame(self.surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            self.surface.present()



class TcnSlangRenderOp(Operator):
    def __init__(self, fragment, stop_cond, *args, **kwargs):
        self.stop_cond = stop_cond
        self.window = None
        self.ui_loop = None
        self.is_closing = False
        self.renderables_config = None
        super().__init__(fragment, stop_cond, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input", size=IOSpec.ANY_SIZE)
        spec.input("input_specs").condition(ConditionType.NONE)
        spec.param("renderables", flag=hs.core.ParameterFlag.NONE)

    def start(self):

        def close_handler():
            self.is_closing = True
            log.info("Close Handler in TcnSlangRenderOp called")
            self.stop_cond.disable_tick()

        self.window = SlangWindow(1024, 768, "TCN RenderTest", close_callback=close_handler)
        if self.renderables:
            self.renderables_config = json.loads(self.renderables)
        self.ui_loop = threading.Thread(target=self.window.run)
        self.ui_loop.start()

    def compute(self, op_input, op_output, context):
        input_spec = op_input.receive("input_specs")
        if input_spec is not None: # make sure it a string ... or later using schema-based serialization like Holoviz
            # probably needs more than just deserializing
            self.renderables_config = json.loads(input_spec)

        value_vector = op_input.receive("input")
        render_context = {}
        if value_vector is not None:
            log.debug(f"TcnSlangRenderOp received input {self.name} {len(value_vector)}")
            for i, value in enumerate(value_vector):
                for k,v in value.items():
                    render_context[k] = v

        def extract_inputs(ctx, mappings):
            kwargs = {}
            if mappings is None:
                return kwargs

            for port_name, buffer_name in mappings:
                buffer = ctx.get(port_name)

                # XXX why is this needed!
                # first attempt to forward a cupy array and do the d2d copy in renderable-update
                if getattr(buffer, "device", None) is not None:
                    # buffer = cp.asnumpy(buffer)
                    buffer = cp.asarray(buffer)

                if buffer is not None:
                    # @FIXME: this downloads the buffer potentially being already on the device in cuda memory.
                    # a more sophisticated resource sharing cuda-vulkan interop should be used and is available in slang
                    kwargs[buffer_name] = buffer
                else:
                    log.warning(f"TcnSlangRenderOp: missing input buffer for {port_name}")
            return kwargs


        for name, config in self.renderables_config.items():  # XX consider priorities here
            renderable = self.window.get_renderable(name)
            if renderable is None:
                log.info(f"TcnSlangRenderOp: create renderable: {name}: {config}")
                # XXX make renderer configurable
                if config["entity_type"] == "pointcloud":
                    kwargs = extract_inputs(render_context, config["input_mappings"])
                    renderable = Pointcloud(device=self.window.get_device(), **kwargs)
                    self.window.add_renderable(name, renderable)
                else:
                    log.warning(f"TcnSlangRenderOp: unsupported entity type: {config['entity_type']}")
            else:
                kwargs = extract_inputs(render_context, config["input_mappings"])
                renderable.update(**kwargs)


        self.window.request_redraw()

    def stop(self):
        if not self.is_closing:
            self.window.close()
        self.ui_loop.join()


class App(hs.core.Application):
    def compose(self):

        print("Starting TCN RenderTest")
        stop_cond = BooleanCondition(self, name="stop_cond")

        input_subgraph = ShmSimpleBackprojectionSubgraph(self, "sbs", fuse_buffers=False)

        # @todo: create renderables config interface once stabilized
        renderables = {
            "camera01_pointcloud": {
                "entity_type": "pointcloud",
                "entity_args": None,
                "renderer": "colored_pointcloud",
                "priority": 0,
                "pose": None,
                "input_mappings": [
                    ("camera01_colorimage", "image"),
                    ("camera01_positions", "positions"),
                    ("camera01_texcoords", "texcoords"),
                ]
            },
            # "camera02_pointcloud": {
            #     "entity_type": "pointcloud",
            #     "entity_args": None,
            #     "renderer": "colored_pointcloud",
            #     "priority": 0,
            #     "pose": None,
            #     "input_mappings": [
            #         ("camera02_colorimage", "image"),
            #         ("camera02_positions", "positions"),
            #         ("camera02_texcoords", "texcoords"),
            #     ]
            # },
            # "camera03_pointcloud": {
            #     "entity_type": "pointcloud",
            #     "entity_args": None,
            #     "renderer": "colored_pointcloud",
            #     "priority": 0,
            #     "pose": None,
            #     "input_mappings": [
            #         ("camera03_colorimage", "image"),
            #         ("camera03_positions", "positions"),
            #         ("camera03_texcoords", "texcoords"),
            #     ]
            # },
            # "camera04_pointcloud": {
            #     "entity_type": "pointcloud",
            #     "entity_args": None,
            #     "renderer": "colored_pointcloud",
            #     "priority": 0,
            #     "pose": None,
            #     "input_mappings": [
            #         ("camera04_colorimage", "image"),
            #         ("camera04_positions", "positions"),
            #         ("camera04_texcoords", "texcoords"),
            #     ]
            # },
            # # "camera04_colorimage_origin": {
            #     "entity_type": "colored_mesh",
            #     "entity_args": ["axis3d",],
            #     "renderer": "colored_mesh",
            #     "priority": 0,
            #     "pose": ["world_origin", "camera01_colorimage"],
            #     "input_mappings": None
            # },
        }


        slang = TcnSlangRenderOp(self, stop_cond, name="Slang", renderables=json.dumps(renderables))
        self.add_flow(input_subgraph, slang, {
            ("color_outputs", "input"),
            ("depth_outputs", "input"),
            ("position_outputs", "input"),
            ("texcoord_outputs", "input"),
        })

        # Visualize the application graph
        #output_path = Path(__file__).parent / "holoscan_graph.gexf"
        #visualize_holoscan_graph(self, output_file=str(output_path), show=False)




def main(config_file=None):
    # make configurable or use holoscan debug level here too
    configure_debug = True

    if configure_debug:
        logging.basicConfig(level=logging.INFO)
        set_log_level(LogLevel.INFO)
        iox2.set_log_level(iox2.LogLevel.Info)
    else:
        logging.basicConfig(level=logging.INFO)
        set_log_level(LogLevel.INFO)

    app = App()
    app.config(config_file)

    if False:
        scheduler = GreedyScheduler(app, name="gs", stop_on_deadlock=True)
    else:
        scheduler = EventBasedScheduler(app, worker_thread_number=24, name="ebs")

    app.scheduler(scheduler)

    with Tracker(app,
                 num_start_messages_to_skip=15,
                 num_last_messages_to_discard=15) as tracker:
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        tracker.print()


if __name__ == "__main__":

    parser = ArgumentParser(description="ARTEKMED Holoscan Rendering Tests.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = config_file = os.path.join(os.path.dirname(__file__), "tcn_render_tests.yaml")
    else:
        config_file = args.config

    main(config_file=config_file)
