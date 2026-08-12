"""LangSAM: open-vocabulary segmentation (Grounding DINO + SAM 2) as reusable sub-flows.

Two variants, sharing model wrappers, prompt/class mapping and panoptic packing:

- **realtime** (`RealtimeLangSamSubgraph`) -- the optimised multi-camera path. Grounding DINO and the
  SAM encoder run as prebuilt TensorRT engines, batched per GPU worker, with a fused CUDA panoptic
  paint. Prompts are baked into the engines at export time, so changing them means re-exporting.
- **prompted** (`PromptedLangSamSubgraph`) -- single-camera PyTorch path. Prompts can change at
  runtime (`TextPromptPublisher`), at a substantially lower frame rate.

Pick by what the application needs to vary: a fixed vocabulary at speed, or a vocabulary the operator
can retype while it runs. They are separate sub-flows rather than one switch because the engine-batch
planning that makes the realtime path fast is precisely what makes its prompts static.

Everything is imported lazily: touching this package must not construct a TensorRT stack, since the
prompted path does not need one and the engine builders import the helpers without any model.
"""

__all__ = [
    # sub-flows
    "RealtimeLangSamSubgraph",
    "PromptedLangSamSubgraph",
    "SingleCameraLangSamSubgraph",
    # realtime per-stage operators, exported so an application can wire them directly
    "GdinoOp",
    "SamOp",
    "PanopticOp",
    "LangSamBatchOp",
    "MaskCollectorOp",
    "LabelMapColorizeOp",
    # prompted path
    "TextPromptPublisher",
    # Single-camera PyTorch reference path, superseded for multi-camera use by
    # PromptedLangSamSubgraph; kept because it is the simplest thing that runs one camera.
    "LangSAM2Operator",
    "LangSamPostprocessorOp",
    # models and helpers, used by the offline engine builders as well as the operators
    "SAM",
    "GDINO",
    "SamTrtEncoder",
    "GDinoTrtDetector",
    "build_panoptic_lut",
    "build_panoptic_map",
    "build_panoptic_map_auto",
    "class_id_map",
    "mask_name",
    "resolve_workers",
    "worker_batch",
    "worker_engine_path",
    "validate_source_cameras",
    # debug / gating
    "MaskDumpOp",
]

_LAZY = {
    "RealtimeLangSamSubgraph": (".realtime", "RealtimeLangSamSubgraph"),
    "GdinoOp": (".realtime_ops", "GdinoOp"),
    "SamOp": (".realtime_ops", "SamOp"),
    "PanopticOp": (".realtime_ops", "PanopticOp"),
    "LangSamBatchOp": (".realtime", "LangSamBatchOp"),
    "MaskCollectorOp": (".realtime", "MaskCollectorOp"),
    "LabelMapColorizeOp": (".realtime", "LabelMapColorizeOp"),
    "PromptedLangSamSubgraph": (".prompted", "PromptedLangSamSubgraph"),
    "SingleCameraLangSamSubgraph": (".single_camera", "SingleCameraLangSamSubgraph"),
    "LangSAM2Operator": (".prompted_ops", "LangSAM2Operator"),
    "TextPromptPublisher": (".prompted_ops", "TextPromptPublisher"),
    "LangSamPostprocessorOp": (".prompted_ops", "LangSamPostprocessorOp"),
    "SAM": (".models", "SAM"),
    "GDINO": (".models", "GDINO"),
    "SamTrtEncoder": (".models", "SamTrtEncoder"),
    "GDinoTrtDetector": (".models", "GDinoTrtDetector"),
    "build_panoptic_lut": (".models", "build_panoptic_lut"),
    "build_panoptic_map_auto": (".models", "build_panoptic_map_auto"),
    "build_panoptic_map": (".helpers", "build_panoptic_map"),
    "class_id_map": (".helpers", "class_id_map"),
    "mask_name": (".helpers", "mask_name"),
    "resolve_workers": (".helpers", "resolve_workers"),
    "worker_batch": (".helpers", "worker_batch"),
    "worker_engine_path": (".helpers", "worker_engine_path"),
    "validate_source_cameras": (".helpers", "validate_source_cameras"),
    "MaskDumpOp": (".mask_dump", "MaskDumpOp"),
}


def __getattr__(name):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module
    return getattr(import_module(module_name, __name__), attr)


def __dir__():
    return sorted(__all__)
