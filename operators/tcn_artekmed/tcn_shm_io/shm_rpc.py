import re
import ctypes
import logging
import time
import iceoryx2 as iox2
from typing import Any, Callable
import numpy as np

from .shm_serde import shm_parameter_rpc as rpc_types

log = logging.getLogger(__name__)

def write_payload(payload, content: memoryview):

    # Create a ctypes array from the raw memory address
    c_array = (ctypes.c_uint8 * payload.len()).from_address(payload.as_ptr())

    # Wrap into a NumPy array (zero-copy view)
    dst_array = np.frombuffer(c_array, dtype=np.uint8)

    # Efficiently copy payload_bytes into the preallocated memory
    src_array = np.frombuffer(content, dtype=np.uint8)
    np.copyto(dst_array, src_array)

def read_payload(payload):
    c_array = (ctypes.c_uint8 * payload.len()).from_address(payload.as_ptr())

    # Wrap into a NumPy array (zero-copy view)
    dst_array = np.frombuffer(c_array, dtype=np.uint8)
    return dst_array


class ParameterRpcServer:

    def __init__(self, node: Any, rpc_service_name: str, fragment: Any):
        self.should_stop_ = False
        self.node_ = node
        self.fragment_ = fragment
        self.node_schema = {}
        self.rpc_service_name_ = rpc_service_name
        self.service_ = (
            node.service_builder(iox2.ServiceName.new(rpc_service_name))
            .request_response(iox2.Slice[ctypes.c_uint8], iox2.Slice[ctypes.c_uint8])
            .open_or_create()
        )
        log.info(f"created RPC Service: {rpc_service_name} ")
        self.server_ = (
            self.service_.server_builder()
            # We guess that the samples are at most 16 bytes in size.
            # This is just a hint to the underlying allocator and is purely optional
            # The better the guess is the less reallocations will be performed
            .initial_max_slice_len(16)
            # The underlying sample size will be increased with a power of two strategy
            # when `ActiveRequest::loan_slice()` or `ActiveRequest::loan_slice_uninit()`
            # requires more memory than available.
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo).create()
        )

    def make_custom_value_type(self, node_name: str, node_class: str, arg_name: str, arg_type: str):

        # rpc_types.ParameterValueType.pvtStruct

        log.info(f"In node {node_name}({node_class}): found arg {arg_name} with unhandled type: {arg_type} - skipped.")
        return None

    def update_schema_from_fragment(self):
        if self.fragment_ is not None:
            graph = self.fragment_.graph
            if graph is not None:
                nodes = graph.get_nodes()
                for node in nodes:
                    parameters = []
                    for arg in node.args:
                        type_str = str(arg.arg_type.to_string)
                        value_type = None

                        if type_str in ["int8_t", "int16_t", "int32_t", "uint8_t", "uint16_t", "uint32_t"]:
                            value_type = rpc_types.ParameterValueType.pvtInt32
                        elif type_str in ["int64_t", "uint64_t"]:
                            value_type = rpc_types.ParameterValueType.pvtInt64
                        elif type_str in ["float"]:
                            value_type = rpc_types.ParameterValueType.pvtFloat
                        elif type_str in ["double"]:
                            value_type = rpc_types.ParameterValueType.pvtDouble
                        elif type_str in ["bool"]:
                            value_type = rpc_types.ParameterValueType.pvtBool
                        elif type_str in ["std::string"]:
                            value_type = rpc_types.ParameterValueType.pvtString
                        else:
                            value_type = self.make_custom_value_type(node.name, type(node).__name__, arg.name, type_str)

                        if value_type is None:
                            continue

                        parameters.append({
                            "key": str(arg.name),
                            "valueType": value_type,
                            "validEntries": [],
                        })
                    if len(parameters) == 0:
                        continue

                    self.node_schema[node.name] = parameters

    def list_nodes(self):
        return list(sorted(self.node_schema.keys()))

    def get_parameter_schema(self, entity_name: str):
        return self.node_schema.get(entity_name, [])

    def handle_request(self, request):
        if request.commandType == rpc_types.RPCCommandType.rpcCommandListEntities:
            response_message = rpc_types.ParameterRpcResponse.new_message(
                commandType=request.commandType,
                responseType=rpc_types.RPCResponseStatus.rpcStatusSuccess,
                entitiesList=self.list_nodes()
            )
            return response_message
        elif request.commandType == rpc_types.RPCCommandType.rpcCommandGetParameterSchema:
            parameters = []
            for item in self.get_parameter_schema(request.entityName):
                parameters.append(rpc_types.ParameterSchema.new_message(**item))
            parameter_schema = rpc_types.ParameterListSchema.new_message(parameters=parameters)

            response_message = rpc_types.ParameterRpcResponse.new_message(
                commandType=request.commandType,
                responseType=rpc_types.RPCResponseStatus.rpcStatusSuccess,
                parameterSchema=parameter_schema
            )
            return response_message


        return None

    def shutdown(self):
        self.should_stop_ = True

    def serve_blocking(self):
        log.info("SHM Parameter RPC Server starting to serve requests.")
        cycle_time = iox2.Duration.from_millis(5)
        try:
            while True and not self.should_stop_:
                self.node_.wait(cycle_time)
                while True and not self.should_stop_:
                    active_request = self.server_.receive()
                    if active_request is not None:
                        data = read_payload(active_request.payload())
                        data_view = memoryview(data)
                        response_bytes = None
                        command_type = None
                        with rpc_types.ParameterRpcRequest.from_bytes(data_view) as request_message:
                            log.info(f"handle request: {request_message.to_dict()}")
                            command_type = request_message.commandType
                            response = self.handle_request(request_message)
                            log.info(f"  send response: {response.to_dict()}")
                            response_bytes = response.to_bytes()

                        if response_bytes is not None:
                            response_bytes_view = memoryview(response_bytes)
                            required_memory_size = len(response_bytes)
                            response = active_request.loan_slice_uninit(required_memory_size)
                            write_payload(response.payload(), response_bytes_view)
                            response = response.assume_init()
                            response.send()
                        else:
                            error_message = rpc_types.RpcParameterResponse.new_message(
                                commandType=command_type,
                                responseType=rpc_types.RPCResponseStatus.rpcStatusError
                            )
                            error_message_bytes = error_message.to_bytes()
                            error_message_bytes_view = memoryview(error_message_bytes)
                            response = active_request.loan_slice_uninit(len(error_message_bytes))
                            write_payload(response.payload(), error_message_bytes_view)
                            response = response.assume_init()
                            response.send()

                    else:
                        break

        except iox2.NodeWaitFailure:
            print("exit")



class ParameterRpcClient:

    def __init__(self, node: Any, rpc_service_name: str):
        self.node_ = node
        self.rpc_service_name_ = rpc_service_name
        self.service_ = (
            node.service_builder(iox2.ServiceName.new(rpc_service_name))
            .request_response(iox2.Slice[ctypes.c_uint8], iox2.Slice[ctypes.c_uint8])
            .open_or_create()
        )
        log.info(f"created RPC Service: {rpc_service_name} ")
        self.client_ = (
            self.service_.client_builder()
            # We guess that the samples are at most 16 bytes in size.
            # This is just a hint to the underlying allocator and is purely optional
            # The better the guess is the less reallocations will be performed
            .initial_max_slice_len(16)
            # The underlying sample size will be increased with a power of two strategy
            # when `ActiveRequest::loan_slice()` or `ActiveRequest::loan_slice_uninit()`
            # requires more memory than available.
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo).create()
        )

    def _call_remote(self, payload: rpc_types.ParameterRpcRequest, response_handler: Callable, timeout_ms: int = 1000):
        payload_bytes = payload.to_bytes()
        payload_view = memoryview(payload_bytes)
        request = self.client_.loan_slice_uninit(len(payload_bytes))
        write_payload(request.payload(), payload_view)
        request = request.assume_init()
        pending_response = request.send()
        # @todo: implement timeout
        start_time = time.time()
        while True:
            response = pending_response.receive()
            if response is not None:
                response_bytes = read_payload(response.payload())
                b = memoryview(response_bytes)  # ensure contiguous bytes
                # Unpacked encoding (most common for "raw message bytes" unless explicitly packed)
                with rpc_types.ParameterRpcResponse.from_bytes(b) as response:
                    return response_handler(response)

            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= timeout_ms:
                log.warning(f"RPC call timed out after {elapsed_ms:.1f}ms")
                break

        return None

    def list_components(self):
        request = rpc_types.ParameterRpcRequest.new_message(commandType=rpc_types.RPCCommandType.rpcCommandListEntities)
        def response_handler(response):
            assert(response.commandType == request.commandType)
            assert(response.which()=="entitiesList")
            return [v for v in response.entitiesList]
        return self._call_remote(request, response_handler)

    def get_parameter_schema(self, entity_name: str):
        request = rpc_types.ParameterRpcRequest.new_message(
            commandType=rpc_types.RPCCommandType.rpcCommandGetParameterSchema,
            entityName=entity_name
        )
        def response_handler(response):
            assert(response.commandType == request.commandType)
            assert(response.which()=="parameterSchema")
            return [v.to_dict() for v in response.parameterSchema.parameters]
        return self._call_remote(request, response_handler)



