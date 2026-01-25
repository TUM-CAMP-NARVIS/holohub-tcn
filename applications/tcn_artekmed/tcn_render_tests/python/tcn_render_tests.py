#
# Place the license header here
#
import logging
import os
from argparse import ArgumentParser
import json

import iceoryx2 as iox2
import holoscan as hs
import cupy as cp
import numpy as np
import json

from holoscan.core import Operator, OperatorSpec, Tracker
from holoscan.conditions import CountCondition, PeriodicCondition, BooleanCondition
from holoscan.logger import LogLevel, set_log_level
from holoscan.operators import HolovizOp, holoviz
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler
from holoscan.resources import CudaStreamPool, BlockMemoryPool, MemoryStorageType, RMMAllocator
from holoscan.pose_tree import PoseTreeManager, SO3, Pose3

from operators.tcn_artekmed.tcn_processing import ShmSimpleBackprojectionSubgraph

from operators.tcn_artekmed.tcn_util.helpers import pose3_to_matrix4x4
from operators.tcn_artekmed.tcn_slang_renderer import TcnSlangRenderOp

#from graph_visualizer import visualize_holoscan_graph


log = logging.getLogger(__name__)


class SingleGofRecorder(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input", size=hs.core.IOSpec.ANY_SIZE)
        spec.param("storage_directory", flag=hs.core.ParameterFlag.NONE)

    def compute(self, op_input, op_output, context):
        inputs = op_input.receive("input")
        log.info(f"SingleGofRecorder received input {self.name}, receivers: {inputs}")
        if not os.path.isdir(self.storage_directory):
            os.mkdir(self.storage_directory)
        metadata = []
        for i, input in enumerate(inputs):
            for tensor_name, tensor in input.items():
                if tensor is None:
                    continue
                log.info(f"Saving tensor {tensor_name} to {self.storage_directory}")
                metadata.append({"name": tensor_name, "shape": tensor.shape})
                cu_array = cp.asarray(tensor)
                np.save(os.path.join(self.storage_directory, f"{tensor_name}.npy"), cp.asnumpy(cu_array))
        json.dump(metadata, open(os.path.join(self.storage_directory, "metadata.json"), "w"))




class SingleGofPlayer(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")
        spec.param("storage_directory", flag=hs.core.ParameterFlag.NONE)

    def compute(self, op_input, op_output, context):

        md_file = os.path.join(self.storage_directory, "metadata.json")
        if not os.path.isfile(md_file):
            log.error(f"No metadata file found at {md_file}")
            return
        with open(md_file, "r") as f:
            metadata = json.load(f)

        message = {}
        for m in metadata:
            tensor_name = m["name"]
            tensor_shape = m["shape"]
            tensor_file = os.path.join(self.storage_directory, f"{tensor_name}.npy")
            tensor = np.load(tensor_file)
            cu_tensor = cp.asarray(tensor)
            message[tensor_name] = cu_tensor
        op_output.emit(message, "")




class App(hs.core.Application):

    def __init__(self, record_type=None, source=None):
        super().__init__()
        self.record_type = record_type
        self.source = source

    def compose(self):

        print("Starting TCN RenderTest")
        stop_cond = BooleanCondition(self, name="stop_cond")

        camera_streams = None
        if self.source == "single_gof":
            pcond = PeriodicCondition(self, 1000, name="pcond")
            camera_streams = SingleGofPlayer(self, pcond, name="sgof_plr", storage_directory="/tmp/sgof_rec")
        elif self.source == "live":
            camera_streams = ShmSimpleBackprojectionSubgraph(self, "sbs", fuse_buffers=False)
        else:
            raise ValueError(f"Invalid source: {self.source}")

        # # @todo: create renderables config interface once stabilized
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

        if self.source == "single_gof":
            self.add_flow(camera_streams, slang, {
                ("output", "input"),
            })
        elif self.source == "live":
            self.add_flow(camera_streams, slang, {
                ("color_outputs", "input"),
                ("depth_outputs", "input"),
                ("position_outputs", "input"),
                ("texcoord_outputs", "input"),
            })

        if self.record_type == "input" and self.source == "live":
            ccond = CountCondition(self, count=1, name="ccond")
            sgof_rec = SingleGofRecorder(self, ccond, name="sgof_rec", storage_directory="/tmp/sgof_rec")
            self.add_flow(camera_streams, sgof_rec, {
                ("color_outputs", "input"),
                ("depth_outputs", "input"),
                ("position_outputs", "input"),
                ("texcoord_outputs", "input"),
            })
        else:
            log.warning("Cannot record streams.")

        # Visualize the application graph
        #output_path = Path(__file__).parent / "holoscan_graph.gexf"
        #visualize_holoscan_graph(self, output_file=str(output_path), show=False)




def main():

    parser = ArgumentParser(description="ARTEKMED Holoscan Rendering Tests.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-r",
        "--record_type",
        choices=["none", "input"],
        default="none",
        help="The stream to record (default: %(default)s).",
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=[
            "live",
            "single_gof",
        ],
        default="live",
        help=(
            "Source of data streams (default: %(default)s)."
        ),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = config_file = os.path.join(os.path.dirname(__file__), "tcn_render_tests.yaml")
    else:
        config_file = args.config


    # make configurable or use holoscan debug level here too
    configure_debug = True

    if configure_debug:
        logging.basicConfig(level=logging.INFO)
        set_log_level(LogLevel.INFO)
        iox2.set_log_level(iox2.LogLevel.Info)
    else:
        logging.basicConfig(level=logging.INFO)
        set_log_level(LogLevel.INFO)

    app = App(source=args.source, record_type=args.record_type)
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
    main()
