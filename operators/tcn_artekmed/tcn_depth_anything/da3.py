import logging
import os
from argparse import ArgumentParser
import threading
import iceoryx2 as iox2

import numpy as np
import cupy as cp
import cv2

import holoscan as hs
from holoscan.core import Operator, OperatorSpec, Subgraph
from holoscan.operators import HolovizOp
from holoscan.resources import UnboundedAllocator
from holoscan.operators import (
    FormatConverterOp,
    InferenceOp,
)

from operators.tcn_artekmed.tcn_util import ConvertBgraToRgbaOp

log = logging.getLogger(__name__)


class DA3PostprocessorOp(Operator):
    """Operator that does postprocessing before sending resulting image to Holoviz"""

    def __init__(self, *args, depth_near=0.3, depth_far=10.0,
                 metric_focal=None, metric_scale_factor=300.0, **kwargs):
        # Metric depth display range in meters. Depth is colorized against this fixed
        # range (not per-frame min/max) so colors stay stable and metric-meaningful.
        # DA3 outputs focal-normalized depth: near = small value, far = large value.
        self.depth_near = depth_near
        self.depth_far = depth_far
        # Metric conversion (DA3 apply_metric_scaling convention):
        #   metric_depth[m] = raw_depth * (focal / scale_factor)
        # where `focal` = (fx + fy) / 2 in pixels at the model input resolution and
        # `scale_factor` is the model's canonical constant (300.0). If metric_focal is
        # None, the raw output is used as-is (no camera intrinsics available).
        self.metric_focal = metric_focal
        self.metric_scale_factor = metric_scale_factor
        super().__init__(*args, **kwargs)
        #
        self.image_dim = 518
        self.mouse_pressed = False
        self.display_modes = ["original", "depth", "side-by-side", "interactive"]
        self.idx = 1
        self.current_display_mode = self.display_modes[self.idx]
        # In interactive mode, how much of the original video to show
        self.ratio = 0.5
        # Throttle for the per-frame metric depth min/max log (used to tune depth_near/far).
        self._frame_count = 0
        self._log_every = 30

    def setup(self, spec: OperatorSpec):
        """
        input:  "input_depthmap"  - Input tensors representing depthmap from inference
        input:  "input_image"     - Input tensor representing the RGB image
        output: "output_image"    - The image for Holoviz to display
        output: "output_specs"    - Text to show the current display mode

        This operator's output image depends on the current display mode, if set to

            * "original": output the original image from input source
            * "depth": output the color depthmap based on the depthmap returned from
                       Depth Anything V2 model
            * "side-by-side": output a side-by-side view of the original image next to
                              the color depthmap
            * "interactive": allow user to control how much of the image to show as
                             original while the rest shows the color depthmap

        Returns:
            None
        """
        spec.input("input_depthmap")
        spec.input("input_image")
        spec.output("output_image")
        spec.output("output_specs")

    def clamp(self, value, min_value=0, max_value=1):
        """Clamp value between [min_value, max_value]"""
        return max(min_value, min(max_value, value))

    def toggle_display_mode(self, *args):
        mouse_button = args[0]
        action = args[1]

        LEFT_BUTTON = 0
        PRESSED = 0

        # If event is for the middle or right mouse button, update some values for interactive mode
        #   - update the status of whether the button is being pressed or released
        #   - update the ratio of the original image to display
        if mouse_button.value != LEFT_BUTTON:
            self.mouse_pressed = action.value == PRESSED
            self.x = self.clamp(self.x, 0, self.framebuffer_size)
            self.ratio = self.x / self.framebuffer_size
            return

        # When left mouse button is pressed, update the display mode
        if action.value == PRESSED:
            self.idx = (self.idx + 1) % len(self.display_modes)
            self.current_display_mode = self.display_modes[self.idx]

    # Update cursor position which will be used in interactive mode
    def cursor_pos_callback(self, *args):
        self.x = args[0]
        if self.mouse_pressed:
            self.x = self.clamp(self.x, 0, self.framebuffer_size)
            self.ratio = self.x / self.framebuffer_size

    # Update size of holoviz framer buffer which will be used to calculate self.ratio
    def framebuffer_size_callback(self, *args):
        self.framebuffer_size = args[0]

    def to_metric(self, depth_map):
        # Convert the model's focal-normalized output to metric depth in meters using the
        # camera focal length. If no focal is available, return the raw output unchanged.
        if self.metric_focal is not None:
            return depth_map * (self.metric_focal / self.metric_scale_factor)
        return depth_map

    def normalize(self, depth_map):
        # Colorize metric depth against the fixed [depth_near, depth_far] range (meters).
        metric = self.to_metric(depth_map)
        # Output feeds COLORMAP_JET (0 -> blue, 255 -> red), so map:
        #   near depth -> 255 -> red,  far depth -> 0 -> blue.
        normalized = (metric - self.depth_near) / (self.depth_far - self.depth_near)
        normalized = cp.clip(normalized, 0.0, 1.0)
        return 255 - (normalized * 255)

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("input_depthmap")
        in_image = op_input.receive("input_image")

        # Convert input to cupy array
        inference_output = cp.asarray(in_message.get("inference_output")).squeeze()

        image = cp.asarray(in_image.get("preprocessed"))

        # Log per-frame metric depth min/max/mean to help tune depth_near/depth_far.
        self._frame_count += 1
        if self._frame_count % self._log_every == 0:
            metric = self.to_metric(inference_output)
            units = "m" if self.metric_focal is not None else "raw"
            log.info(
                f"DA3 metric depth [{units}]: min={float(cp.min(metric)):.3f} "
                f"max={float(cp.max(metric)):.3f} mean={float(cp.mean(metric)):.3f} "
                f"(display range near={self.depth_near} far={self.depth_far})"
            )

        if self.current_display_mode == "original":
            # Display the original image
            image = (image * 255).astype(cp.uint8)
            output_image = image
        elif self.current_display_mode == "depth":
            # Display the color depthmap
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            output_image = depth_colormap

        elif self.current_display_mode == "side-by-side":
            # Display both original and color depthmap images side-by-side
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            image = (image * 255).astype(cp.uint8)
            output_image = cp.hstack((image, depth_colormap))
        else:
            # Interactive mode
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            image = (image * 255).astype(cp.uint8)
            pos = int(self.image_dim * self.ratio)
            output_image = cp.hstack(
                (
                    image[:, :pos, :],
                    depth_colormap[
                        :,
                        pos:,
                    ],
                )
            )

        # Position display mode text near bottom left corner of Holoviz window
        display_mode_text = np.asarray([(0.025, 0.9)])

        # Create output message
        out_message = {"display_mode": display_mode_text, "image": hs.as_tensor(output_image)}
        op_output.emit(out_message, "output_image")

        # holoviz specs for displaying the current display mode
        specs = []
        spec = HolovizOp.InputSpec("display_mode", "text")
        spec.text = [self.current_display_mode]
        spec.color = [1.0, 1.0, 1.0, 1.0]
        spec.priority = 1
        specs.append(spec)
        op_output.emit(specs, "output_specs")



class DA3MetricProcessingSubgraph(Subgraph):
    """Subgraph containing the shm-receiver and backprojection pipeline."""

    def __init__(self, fragment, name, kwargs, device_context=None):
        self.kwargs = kwargs
        self.device_context = device_context
        super().__init__(fragment, name)

    def _make_name(self, name):
        return f"{self.name}_{name}"

    def _compute_metric_focal(self, proc_w, proc_h):
        """Focal length (px) of the color camera scaled to the model input resolution.

        Returns None when no device_context is available, in which case the
        postprocessor falls back to using the raw (focal-normalized) output.
        """
        if self.device_context is None:
            log.warning("DA3: no device_context provided; metric scaling disabled (raw depth)")
            return None
        try:
            color = self.device_context["calibration"]["colorCameraParameters"]
            fx = float(color["fovX"])  # focal_length.x in px (original color resolution)
            fy = float(color["fovY"])  # focal_length.y in px
            color_w = float(color["width"])
            color_h = float(color["height"])
        except (KeyError, TypeError, ValueError) as e:
            log.warning(f"DA3: could not read color intrinsics from device_context ({e}); "
                        "metric scaling disabled (raw depth)")
            return None

        fx_scaled = fx * (proc_w / color_w)
        fy_scaled = fy * (proc_h / color_h)
        focal = (fx_scaled + fy_scaled) / 2.0
        log.info(
            f"DA3 metric scaling: focal={focal:.2f}px @ {int(proc_w)}x{int(proc_h)} "
            f"(color fx={fx:.1f}, fy={fy:.1f} @ {int(color_w)}x{int(color_h)})"
        )
        return focal

    def compose(self):
        log.info("Compose subgraph: DA3MetricProcessing")
        app = self.fragment.application

        # @todo: do not use app.kwargs directly, but pass the relevant dictionary or subtree to the subgraph explicitly

        # format converter only supports rgba not bgra ..
        in_dtype = "rgba8888"
        pool = UnboundedAllocator(self, name="pool")
        da3_preprocessor_args = self.kwargs("da3_preprocessor")
        da3_preprocessor = FormatConverterOp(
        self,
            name=self._make_name("da3_preprocessor"),
            pool=pool,
            in_dtype=in_dtype,
            **da3_preprocessor_args,
        )

        da3_inference_args = self.kwargs("da3_inference")
        da3_inference_config = self.kwargs("da3_inference_config")
        da3_inference_args["model_path_map"] = {
            "depth_v3": da3_inference_config.get("model_path")
        }

        da3_inference = InferenceOp(
            self,
            name=self._make_name("da3_inference"),
            allocator=pool,
        **da3_inference_args,
        )

        # Derive the focal length (in pixels) at the model input resolution so the
        # postprocessor can convert the model's focal-normalized output to metric depth.
        # FormatConverterOp resizes the color frame to resize_width x resize_height (a plain
        # squash), so the effective focal per axis scales by (proc / original) dimension.
        proc_w = float(da3_preprocessor_args.get("resize_width", 518))
        proc_h = float(da3_preprocessor_args.get("resize_height", 518))
        metric_focal = self._compute_metric_focal(proc_w, proc_h)

        da3_postprocessor = DA3PostprocessorOp(
            self,
            name=self._make_name("da3_postprocessor"),
            allocator=pool,
            depth_near=da3_inference_config.get("depth_near", 0.3),
            depth_far=da3_inference_config.get("depth_far", 10.0),
            metric_focal=metric_focal,
            metric_scale_factor=da3_inference_config.get("metric_scale_factor", 300.0),
        )

        self.add_flow(da3_preprocessor, da3_postprocessor, {("tensor", "input_image")})
        self.add_flow(da3_preprocessor, da3_inference, {("", "receivers")})
        self.add_flow(da3_inference, da3_postprocessor, {("transmitter", "input_depthmap")})

        # Expose the relevant ports
        self.add_input_interface_port("input", da3_preprocessor, "source_video")
        self.add_output_interface_port("output_image", da3_postprocessor, "output_image")
        self.add_output_interface_port("output_specs", da3_postprocessor, "output_specs")
