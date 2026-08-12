import logging

import holoscan as hs
from holoscan.core import Subgraph
from holoscan.resources import UnboundedAllocator

from operators.tcn_artekmed.tcn_util import ConvertBgraToRgbaOp
from .prompted_ops import LangSAM2Operator, LangSamPostprocessorOp, TextPromptPublisher

log = logging.getLogger(__name__)


class PromptedLangSamSubgraph(Subgraph):
    """Subgraph containing LangSAM inference pipeline (Grounding DINO + SAM2)"""

    def __init__(self, fragment, name, allocator, kwargs):
        self.kwargs = kwargs
        self.allocator = allocator
        super().__init__(fragment, name)

    def _make_name(self, name):
        return f"{self.name}_{name}"

    def compose(self):
        log.info("Compose subgraph: LangSamProcessing")

        # Color format converter (BGRA to RGBA)
        col_conv = ConvertBgraToRgbaOp(self,
                                       name=self._make_name("color_converter_rgba"),
                                       allocator=self.allocator)

        # Allocator for operators
        pool = UnboundedAllocator(self, name="pool")

        # Text prompt publisher
        text_prompt_args = self.kwargs("text_prompts")
        text_prompt_publisher = TextPromptPublisher(
            self,
            name=self._make_name("text_prompt_publisher"),
            **text_prompt_args,
        )

        # LangSAM inference operator
        langsam_args = self.kwargs("langsam_inference")
        langsam_inference = LangSAM2Operator(
            self,
            name=self._make_name("langsam_inference"),
            **langsam_args,
        )

        # LangSAM postprocessor for visualization. Pass the same prompt list so the
        # postprocessor can assign a stable color per class (seeded from prompt order).
        langsam_postprocessor_args = self.kwargs("langsam_postprocessor")
        langsam_postprocessor = LangSamPostprocessorOp(
            self,
            name=("langsam_postprocessor"),
            prompts=text_prompt_args.get("prompts"),
            **langsam_postprocessor_args,
        )

        # Connect operators
        self.add_flow(col_conv, langsam_inference, {("output", "image")})
        self.add_flow(text_prompt_publisher, langsam_inference, {("out", "text_prompts")})
        self.add_flow(langsam_inference, langsam_postprocessor, {("out", "in")})

        # Expose interface ports
        self.add_input_interface_port("input", col_conv, "input")
        self.add_output_interface_port("output_masks", langsam_postprocessor, "out")
