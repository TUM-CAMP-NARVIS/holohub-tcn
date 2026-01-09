#
# Place the license header here
#
import logging
import os
from argparse import ArgumentParser
import threading

from typing import Callable, Optional, Union
import slangpy as spy
from pathlib import Path

import numpy as np
import cupy as cp
import holoscan as hs

from holoscan.core import Operator, OperatorSpec, Tracker, ConditionType, IOSpec
from holoscan.conditions import CountCondition, PeriodicCondition, BooleanCondition
from holoscan.logger import LogLevel, set_log_level
from holoscan.operators import HolovizOp, holoviz
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler
from holoscan.resources import CudaStreamPool, BlockMemoryPool, MemoryStorageType, RMMAllocator
from holoscan.pose_tree import PoseTreeManager, SO3, Pose3

log = logging.getLogger(__name__)

from mesh_renderer import MeshRenderer
from mesh_data import Mesh
from pointcloud_renderer import PointcloudRenderer
from pointcloud_data import Pointcloud

class SlangWindow:
    def __init__(self, width: int, height: int, title: str, resizeable: bool = True, close_callback: Callable = None):
        self.ui = None
        self.close_callback = close_callback

        self.window = spy.Window(width, height, title, resizable=resizeable)
        asset_root_dir = Path(__file__).parent / "assets"

        self.device = spy.Device(
            enable_debug_layers=True,
            compiler_options={"include_paths": [str(asset_root_dir / "shaders")]},
        )

        model_path = asset_root_dir / "models" / "monkey.obj"

        self.mesh = Mesh.from_obj(self.device, str(model_path))
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(self.window.width, self.window.height)
        self.mesh_renderer = MeshRenderer(self.device, self.surface.config.format)
        self.window.on_keyboard_event = self._on_window_keyboard_event
        self.window.on_mouse_event = self._on_window_mouse_event
        self.window.on_resize = self.handle_resize
        self.create_depth_texture()
        self.setup_ui()
        self.dirty = False
        self.surface_texture = None
        self._mouse_pos = None

        # Hookable events
        self.on_keyboard_event: Optional[Callable[[spy.KeyboardEvent], None]] = None
        self.on_mouse_event: Optional[Callable[[spy.MouseEvent], None]] = None


    def setup_ui(self):
        self.ui = spy.ui.Context(self.device)

        window = spy.ui.Window(
            self.ui.screen, "Settings", spy.float2(10, 10), spy.float2(300, 100)
        )

        spy.ui.Text(window, "Hello, World!")

    def close(self):
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

    def _on_window_keyboard_event(self, event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.close()
                return
        if self.on_keyboard_event:
            self.on_keyboard_event(event)
        else:
            self.ui.handle_keyboard_event(event)

    def _on_window_mouse_event(self, event: spy.MouseEvent):
        if event.type == spy.MouseEventType.move:
            self._mouse_pos = event.pos
        if self.on_mouse_event:
            self.on_mouse_event(event)
        else:
            self.ui.handle_mouse_event(event)


    def run(self):
        while not self.window.should_close():
            self.window.process_events()
            #self.ui.process_events()

            if self.dirty:
                self.resize()
                self.dirty = False

            self.surface_texture = self.surface.acquire_next_image()

            if not self.surface_texture:
                continue

            command_encoder = self.device.create_command_encoder()
            window_size = (self.window.width, self.window.height)

            self.ui.begin_frame(*window_size)

            self.mesh_renderer.render(
                command_encoder,
                self.mesh,
                window_size,
                self.surface_texture,
                self.depth_texture,
            )

            self.ui.end_frame(self.surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            self.surface.present()


class SourceOp(Operator):
    """
    A simple source operator that generates a single-element Tensor containing incrementing
    integer values.

    This operator serves as a data source in the Holoscan pipeline, emitting
    incrementing integer values starting from 1. Each compute cycle produces
    a new value that gets passed to downstream operators.

    Attributes:
        index (int): The current value to emit, increments after each emission
    """

    def __init__(self, *args, **kwargs):
        self.index = 1
        super().__init__(*args, **kwargs)

    def setup(self, spec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        value = cp.array([self.index], dtype=cp.int32)
        op_output.emit(dict(output=value), "output")
        self.index += 1



class TcnSlangRenderOp(Operator):
    def __init__(self, fragment, stop_cond, *args, **kwargs):
        self.stop_cond = stop_cond
        self.slang_window = None
        self.ui_loop = None
        self.is_closing = False
        super().__init__(fragment, stop_cond, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input", size=IOSpec.ANY_SIZE)
        spec.input("input_specs").condition(ConditionType.NONE)
        spec.param("shader_source_file", flag=hs.core.ParameterFlag.NONE)

    def start(self):

        def close_handler():
            self.is_closing = True
            log.info("Close Handler in TcnSlangRenderOp called")
            self.stop_cond.disable_tick()

        self.slang_window = SlangWindow(1024, 768, "TCN RenderTest", close_callback=close_handler)
        self.ui_loop = threading.Thread(target=self.slang_window.run)
        self.ui_loop.start()

    def compute(self, op_input, op_output, context):
        sig1 = op_input.receive("input")
        log.info(f"TcnSlangRenderOp received input {self.name}")

    def stop(self):
        if not self.is_closing:
            self.slang_window.close()
        self.ui_loop.join()


class App(hs.core.Application):
    def compose(self):

        stop_cond = BooleanCondition(self, name="stop_cond")
        # Add your operators here
        print("Starting TCN RenderTest")
        source = SourceOp(self, name="Source")

        slang = TcnSlangRenderOp(self, stop_cond, name="Slang", shader_source_file="simple.slang")
        # Execute the pipeline10 times
        #slang.add_arg(PeriodicCondition(self, 1000000000))

        self.add_flow(source, slang, {("output", "input")})


def main(config_file=None):
    # make configurable or use holoscan debug level here too
    configure_debug = True

    if configure_debug:
        logging.basicConfig(level=logging.DEBUG)
        set_log_level(LogLevel.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
        set_log_level(LogLevel.INFO)

    app = App()
    app.config(config_file)

    if configure_debug:
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
