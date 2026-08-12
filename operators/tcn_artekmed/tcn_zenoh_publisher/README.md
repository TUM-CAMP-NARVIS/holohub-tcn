# tcn_zenoh_publisher

`TcnZenohPublisherOp` — sink operator that publishes an encoded payload to a Zenoh topic.

## Ports

| port | dir | type |
|---|---|---|
| `input` | in | `std::vector<uint8_t>` — payload to publish |
| `type_name` | in | `std::string`, **no condition** — optional type annotation |

`type_name` carries `ConditionType::kNone`, so the operator publishes whether or not a type name is
supplied on a given tick.

## Parameters

| parameter | notes |
|---|---|
| `topic` | Zenoh key expression to publish on. |

## Usage

The counterpart of `tcn_zenoh_subscriber`. Encoding happens upstream — see `tcn_cdr_serde` for the
serialisation library and type registry.
