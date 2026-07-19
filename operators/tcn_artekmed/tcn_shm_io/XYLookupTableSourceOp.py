import logging
from typing import Any

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec
from .DeviceContext import DeviceContextService

log = logging.getLogger("XYLookupTableSourceOp")

class XYLookupTableSourceOp(Operator):
    def __init__(self, fragment: Any, *args, **kwargs):
        self.ctx_service = None
        self.xy_table_data = None
        super().__init__(fragment, *args, **kwargs)

    def initialize(self):
        self.xy_table_data = cp.asarray(self.ctx_service.get_xy_table(self.camera_name))

    def setup(self, spec: OperatorSpec):
        spec.output("xy_table")
        spec.param("camera_name")
        self.ctx_service = self.service(DeviceContextService)

    def compute(self, op_input, op_output, context):
        if self.xy_table_data is not None:
            try:
                xytable_tensor = hs.as_tensor(self.xy_table_data)
                op_output.emit({"": xytable_tensor}, "xy_table")
            except Exception as e:
                log.exception(e)
        else:
            log.error(f"XYLookupTableSourceOp: Could not create XY Table for camera: {self.camera_name}")
