# main.py
import os
import pathlib
import slint
from typing import Any, List
import logging

import iceoryx2 as iox2
from operators.tcn_artekmed.tcn_shm_io import ParameterRpcClient

log = logging.getLogger(__name__)

MODULE_BASE_PATH = os.path.dirname(__file__)
UI_BASE_PATH = os.path.join(MODULE_BASE_PATH, "ui")
SLINT_LIBRARY_PATHS = {"sleek-ui": pathlib.Path(os.path.join(MODULE_BASE_PATH, "modules", "sleek-ui"))}
app_window = slint.load_file(os.path.join(UI_BASE_PATH, "app_window.slint"), library_paths=SLINT_LIBRARY_PATHS)

class MainWindowApp(app_window.AppWindow):
    def __init__(self):
        self._rpc_client = None
        super().__init__()

    @slint.callback(global_name="RpcClient")
    def check_connectivity(self):
        return self._rpc_client is not None

    @slint.callback(global_name="RpcClient")
    def list_components(self):
        if self._rpc_client is not None:
            return self._rpc_client.list_components()
        return []

    @slint.callback(global_name="RpcClient")
    def get_parameter_schema(self, entity_name: str):
        if self._rpc_client is not None:
            return self._rpc_client.get_parameter_schema(entity_name)
        return []

    @slint.callback(global_name="RpcClient")
    def get_parameter_values(self, entity_name: str):
        if self._rpc_client is not None:
            return self._rpc_client.get_parameter_values(entity_name)
        return []

    @slint.callback(global_name="RpcClient")
    def set_parameter_values(self, entity_name: str, values: List[Any]):
        if self._rpc_client is not None:
            return self._rpc_client.set_parameter_values(entity_name, values)
        return False


    def set_fragment(self, rpc_client: ParameterRpcClient):
        self._rpc_client = rpc_client

        if self._rpc_client is not None:
            component_names = self._rpc_client.list_components()
            self.sidebar_pages = slint.ListModel(component_names)
            if component_names:
                self.sidebar_current_page = component_names[0]

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
