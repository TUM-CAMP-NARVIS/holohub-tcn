# tcn_cdr_decoder

`TcnCdrDecoderOp` — decodes a CDR-encoded message using the type registry.

## Ports

| port | dir | type |
|---|---|---|
| `input` | in | `std::vector<uint8_t>` — raw CDR payload |
| `type_name` | in | `std::string` — registered type name, used to select the decoder |
| `output` | out | `std::vector<uint8_t>` — decoded payload |

## Parameters

| parameter | notes |
|---|---|
| `source_name` | Logical source this decoder belongs to. |
| `stream_index` | Index of the stream within that source. |

## Usage

Sits between `tcn_zenoh_subscriber` and a consumer. The type is resolved at runtime from
`type_name` against the registry in `tcn_cdr_serde`, so an unregistered type is a runtime failure
rather than a compile-time one — register the types the application expects at startup.
