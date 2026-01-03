@0xa16bf610e56092b2;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("tcnart_msgs::rpc");

# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------

enum RPCResponseStatus {
  rpcStatusSuccess @0;
  rpcStatusError   @1;
}

enum ParameterValueType {
  pvtString @0;
  pvtInt32  @1;
  pvtInt64  @2;
  pvtFloat  @3;
  pvtDouble @4;
  pvtBool   @5;
  pvtStruct @6;
}

# ----------------------------------------------------------------------
# Basic “Null” RPC
# ----------------------------------------------------------------------

struct NullRequest {
  dummy @0 :UInt8;
}

struct NullReply {
  status @0 :RPCResponseStatus;
}

# ----------------------------------------------------------------------
# String
# ----------------------------------------------------------------------

struct StringRequest {
  value @0 :Text;
}

struct StringReply {
  value  @0 :Text;
  status @1 :RPCResponseStatus;
}

# ----------------------------------------------------------------------
# Bool
# ----------------------------------------------------------------------

struct BoolRequest {
  value @0 :Bool;
}

struct BoolReply {
  value  @0 :Bool;
  status @1 :RPCResponseStatus;
}

# ----------------------------------------------------------------------
# UInt32
# ----------------------------------------------------------------------

struct UInt32Request {
  value @0 :UInt32;
}

struct UInt32Reply {
  value  @0 :UInt32;
  status @1 :RPCResponseStatus;
}

# ----------------------------------------------------------------------
# UInt64
# ----------------------------------------------------------------------

struct UInt64Request {
  value @0 :UInt64;
}

struct UInt64Reply {
  value  @0 :UInt64;
  status @1 :RPCResponseStatus;
}

# ----------------------------------------------------------------------
# Int32
# ----------------------------------------------------------------------

struct Int32Request {
  value @0 :Int32;
}

struct Int32Reply {
  value  @0 :Int32;
  status @1 :RPCResponseStatus;
}

# ----------------------------------------------------------------------
# Int64
# ----------------------------------------------------------------------

struct Int64Request {
  value @0 :Int64;
}

struct Int64Reply {
  value  @0 :Int64;
  status @1 :RPCResponseStatus;
}

# ----------------------------------------------------------------------
# Float32
# ----------------------------------------------------------------------

struct Float32Request {
  value @0 :Float32;
}

struct Float32Reply {
  value  @0 :Float32;
  status @1 :RPCResponseStatus;
}

# ----------------------------------------------------------------------
# Float64
# ----------------------------------------------------------------------

struct Float64Request {
  value @0 :Float64;
}

struct Float64Reply {
  value  @0 :Float64;
  status @1 :RPCResponseStatus;
}

# ----------------------------------------------------------------------
# String list
# ----------------------------------------------------------------------

struct StringListRequest {
  values @0 :List(Text);
}

struct StringListReply {
  values @0 :List(Text);
  status @1 :RPCResponseStatus;
}

# ----------------------------------------------------------------------
# Generic parameter interfaces
# ----------------------------------------------------------------------

struct ParameterList {
  parameters @0 :List(Parameter);
}

struct Parameter {
  key   @0 :Text;
  value @1 :ParameterValue;
}

struct ParameterValue {
  union {
    stringValue @0 :Text;
    int32Value  @1 :Int32;
    int64Value  @2 :Int64;
    floatValue  @3 :Float32;
    doubleValue @4 :Float64;
    boolValue   @5 :Bool;
    structValue @6 :ParameterList;
  }
}

struct ParameterSchema {
  key          @0 :Text;
  valueType    @1 :ParameterValueType;
  validEntries @2 :List(ParameterValue);
}

struct ParameterListSchema {
  parameters @0 :List(ParameterSchema);
}

# ----------------------------------------------------------------------
# Generic parameter RPCs
# ----------------------------------------------------------------------

struct GenericParameterRequest {
  value @0 :ParameterList;
}

struct GenericParameterReply {
  value  @0 :ParameterList;
  status @1 :RPCResponseStatus;
}

struct ParameterSchemaReply {
  schema @0 :ParameterListSchema;
  status @1 :RPCResponseStatus;
}

enum RPCCommandType {
  rpcCommandListEntities @0;
  rpcCommandGetParameterSchema @1;
  rpcCommandGetParameterValues @2;
  rpcCommandSetParameterValues @3;
}

struct ParameterRpcRequest {
    commandType @0: RPCCommandType;
    entityName @1: Text;
    payload @2: ParameterList;
}

struct ParameterRpcResponse {
    commandType @0: RPCCommandType;
    responseType @1: RPCResponseStatus;
    union {
        entitiesList    @2: List(Text);
        parameterSchema @3: ParameterListSchema;
        parameterValues @4: ParameterList;
    }
}