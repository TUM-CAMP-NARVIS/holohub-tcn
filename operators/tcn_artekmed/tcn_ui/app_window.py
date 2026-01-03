# main.py
import os
import pathlib
import slint
from typing import Any
import logging

import iceoryx2 as iox2
from operators.tcn_artekmed.tcn_shm_io import ParameterRpcClient

log = logging.getLogger(__name__)

MODULE_BASE_PATH = os.path.dirname(__file__)
UI_BASE_PATH = os.path.join(MODULE_BASE_PATH, "ui")
SLINT_LIBRARY_PATHS = {"sleek-ui": pathlib.Path(os.path.join(MODULE_BASE_PATH, "modules", "sleek-ui"))}

class MainWindowApp(slint.load_file(os.path.join(UI_BASE_PATH, "app_window.slint"), library_paths=SLINT_LIBRARY_PATHS).AppWindow):
    def __init__(self):
        self._rpc_client = None
        super().__init__()

    def set_fragment(self, rpc_client: ParameterRpcClient):
        self._rpc_client = rpc_client

        if self._rpc_client is not None:
            for name in self._rpc_client.list_components():
                log.info(f"found node with parameters: {name}")
                for item in self._rpc_client.get_parameter_schema(name):
                    log.info(f"  parameter {item['key']} type: {item['valueType']}")

# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    iox2.set_log_level(iox2.LogLevel.Info)
    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
    rpc_service_name = "tcn_shm_receiver"
    client = ParameterRpcClient(node, f"{rpc_service_name}/PARAMETER_RPC/Components")
    app = MainWindowApp()
    app.set_fragment(client)
    app.run()
