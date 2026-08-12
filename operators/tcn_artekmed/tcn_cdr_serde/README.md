# tcn_cdr_serde

**Library, not an operator.** CDR serialisation/deserialisation and the runtime type registry used by
the Zenoh transport operators.

## Contents

| file | purpose |
|---|---|
| `cdr_serde.hpp` / `.cpp` | CDR encode/decode primitives |
| `cdr_type_registry.hpp` / `.cpp` | Runtime registry mapping a type **name** to its decoder |

## Why a registry

Messages arrive over Zenoh carrying a type name as a string; the decoder is chosen at runtime from
that name. Consequences worth knowing:

- A type the application never registered is a **runtime** failure, not a build error. Register
  everything the deployment expects during startup.
- Publisher and subscriber must agree on names, not just layouts. Renaming a type breaks the wire
  contract even when the bytes are unchanged.

Used by `tcn_cdr_decoder`, `tcn_zenoh_publisher`, `tcn_zenoh_subscriber` and `tcn_zenoh_receiver`.
