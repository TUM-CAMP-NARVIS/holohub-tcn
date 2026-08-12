# tcn_zenoh_subscriber

`TcnZenohSubscriberOp` — source operator that subscribes to a Zenoh topic and emits the raw payload.

## Ports

| port | dir | type |
|---|---|---|
| `output` | out | `std::vector<uint8_t>` — the raw CDR-encoded payload |
| `type_name` | out | `std::string` — the payload's registered type name |

## Parameters

| parameter | notes |
|---|---|
| `topic` | Zenoh key expression to subscribe to. |
| `async_condition` | `AsynchronousCondition` the subscription callback drives, so the operator ticks on arrival rather than polling. |

## Usage

The payload is not decoded here — pair it with `tcn_cdr_decoder`, which uses `type_name` to look up
the type in the registry:

```
tcn_zenoh_subscriber ──output───► tcn_cdr_decoder ──output──► consumer
                     └─type_name─►
```

For image streams, `tcn_zenoh_receiver` does subscription, decode and GPU upload in one operator.
