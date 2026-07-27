import logging
import os
import threading
import json
import time
import math
from dataclasses import dataclass

from typing import Callable, Optional, Union
import slangpy as spy
from pathlib import Path

import numpy as np
import cupy as cp
import holoscan as hs

from holoscan.core import Operator, OperatorSpec, ConditionType, IOSpec
from .slangpy_renderer.renderables import Pointcloud
from .slangpy_renderer.window import SlangWindow

log = logging.getLogger("TcnSlangRenderOp")

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
        log.info("TcnSlangRenderOp::compute")
        input_spec = op_input.receive("input_specs")
        if input_spec is not None: # make sure it a string ... or later using schema-based serialization like Holoviz
            # probably needs more than just deserializing
            self.renderables_config = json.loads(input_spec)

        value_vector = op_input.receive("input")
        port_stream = op_input.receive_cuda_stream("input", False)

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

