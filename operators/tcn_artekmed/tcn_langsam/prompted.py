"""Promptable LangSAM: the optimised multi-camera pipeline with a runtime-editable vocabulary.

Same structure, worker split, batching and SAM/panoptic optimisations as
`RealtimeLangSamSubgraph` -- the *only* differences are that Grounding DINO runs in PyTorch so the
prompt set is not baked into an engine, and that the subgraph exposes a `prompts` input port.

One prompt set drives every camera. That is not a simplification: `GDINO.predict_gpu_batch` encodes
the caption once and repeats it across the image batch, so a shared vocabulary is exactly what keeps
the batched forward batched. Per-camera vocabularies would mean one text encode per camera and would
undo the batching this path exists to keep.

What stays optimised, and why it can:

- **SAM image encoder (TensorRT)** -- its only input is the image. Prompts never enter it.
- **SAM mask decoder / batched decode** -- prompted by *boxes*, not text. Box counts already vary per
  frame, which is what the batched decode pads for.
- **Fused CUDA panoptic paint** -- writes the packed uint16 values it is handed; the class mapping is
  computed host-side from the current prompt list.
- **Per-worker batching and the GPU split** -- batching is over images, not prompts.

So only Grounding DINO's text branch is affected by making the vocabulary dynamic. Nothing about SAM
has to become dynamic, which is what makes this path cost a slower detector rather than a slower
pipeline.

Cost: the PyTorch detector is materially slower than the TensorRT engine, and it re-encodes the
caption on every call. Expect a lower frame rate than `RealtimeLangSamSubgraph`, in the detector
only.

Backend: controlled by `langsam_inference.prompted_gdino_backend`, default `pytorch` (any prompt,
any time). It deliberately does **not** read `gdino_backend` -- a config written for the realtime path
would otherwise silently restrict this sub-flow to the engine's baked vocabulary, and you would only
discover it when an update was rejected.

Setting it to `trt` keeps the fast detector, but then a prompt update may only be a **subset or
reordering of the vocabulary baked into the engine**; anything else is rejected per update and the
pipeline continues on its previous prompts. See `build_prompt_remap`.
"""
import logging

from holoscan.core import Subgraph

from .helpers import resolve_workers
from .realtime import LabelMapColorizeOp, LangSamBatchOp, MaskCollectorOp

log = logging.getLogger(__name__)


class PromptedLangSamSubgraph(Subgraph):
    """Multi-camera LangSAM whose prompt/class definition can be changed while it runs.

    Ports:
        input    -- colour entity carrying every camera's image (workers self-select)
        prompts  -- ``{"text_prompts": [str, ...]}``; conditionless, so frames flow without it

    Outputs `output_masks`, `output_viz` and `output_specs`, identical in shape and meaning to
    `RealtimeLangSamSubgraph`, so a consumer (e.g. the mask/depth join) cannot tell the two apart.
    """

    def __init__(self, fragment, name, kwargs, all_color_cameras):
        self.kwargs = kwargs
        self.all_color_cameras = list(all_color_cameras)
        super().__init__(fragment, name)

    def _n(self, s):
        return f"{self.name}_{s}"

    def _get(self, key):
        try:
            return self.kwargs(key)
        except Exception:
            return None

    def compose(self):
        log.info("Compose subgraph: PromptedLangSam (multi-camera, runtime prompts)")
        multicam_cfg = self._get("gpu_workers") or {}
        langsam_cfg = dict(self.kwargs("langsam_inference"))
        # The backend comes from a DEDICATED key, not from `gdino_backend`. Inheriting that one
        # would let a config written for the realtime path silently restrict this sub-flow to its
        # baked vocabulary -- you would select "prompted", get subset-only prompting, and only find
        # out when an update was rejected. Opting into the fast-but-restricted tier has to be
        # deliberate, so it needs its own key.
        langsam_cfg["gdino_backend"] = str(
            langsam_cfg.get("prompted_gdino_backend", "pytorch")).lower()
        if langsam_cfg["gdino_backend"] not in ("pytorch", "trt"):
            raise ValueError(
                f"langsam_inference.prompted_gdino_backend must be 'pytorch' or 'trt', got "
                f"{langsam_cfg['gdino_backend']!r}")
        if langsam_cfg["gdino_backend"] == "trt":
            log.warning(
                "PromptedLangSam with gdino_backend='trt': prompt updates are restricted to a "
                "subset or reordering of the vocabulary baked into the engine. Anything else is "
                "rejected per update and the previous prompts stay in force.")

        prompts = (self._get("text_prompts") or {}).get("prompts", [])
        workers = resolve_workers(multicam_cfg, self.all_color_cameras)
        log.info(f"PromptedLangSam workers: {workers} (initial prompts: {prompts})")

        collector = MaskCollectorOp(self, name=self._n("collector"))
        colorize = LabelMapColorizeOp(self, name=self._n("colorize"),
                                      num_classes=len(prompts), promptable=True)
        self.add_flow(collector, colorize, {("masks", "masks")})

        # Monolithic workers only. The split (pipelined) variant would need the same prompt update
        # delivered to its gdino and panoptic stages separately and kept consistent between them;
        # pipelining is off by default (it measured slower), so that is unbuilt rather than hidden.
        if bool(multicam_cfg.get("pipelined", False)):
            raise ValueError(
                "PromptedLangSam does not support gpu_workers.pipelined: the prompt update would "
                "have to reach the gdino and panoptic stages separately and stay consistent across "
                "them. Use the monolithic worker (pipelined: false), which is also the faster "
                "configuration.")

        for i, w in enumerate(workers):
            op = LangSamBatchOp(self, name=self._n(f"worker{i}"),
                                cameras=w["cameras"], device=w["device"],
                                langsam_cfg=langsam_cfg, prompts=prompts,
                                promptable=True)
            self.add_flow(op, collector, {("masks", "receivers")})
            # Both interface ports fan out to every worker: one external colour stream and one
            # external prompt stream feed all of them, which is what makes the vocabulary shared.
            self.add_input_interface_port("input", op, "color_input")
            self.add_input_interface_port("prompts", op, "prompts")

        # The colouriser needs the prompt count too, to grow its LUT before a new class id appears
        # in a map it has to index.
        self.add_input_interface_port("prompts", colorize, "prompts")

        self.add_output_interface_port("output_masks", collector, "masks")
        self.add_output_interface_port("output_viz", colorize, "viz")
        self.add_output_interface_port("output_specs", colorize, "specs")
