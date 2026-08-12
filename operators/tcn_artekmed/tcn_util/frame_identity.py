"""Frame identity: acquisition timestamps, and the tensor-map key they add.

Generic to any multi-stage Holoscan pipeline that must track which source frame a message came from
-- nothing here is specific to a model or a camera. It lives in tcn_util because every operator that
forwards frame identity needs it.

The two halves are connected. Attaching an acquisition timestamp adds a `nvidia::gxf::Timestamp`
COMPONENT to the entity, and Holoscan's entity -> dict conversion enumerates every component, not
just the tensors -- so the act of stamping a message is also what puts a non-tensor key into every
downstream consumer's tensor map. `tensor_names()` is the other half of `acq_timestamp()`, not an
unrelated utility.
"""

# --------------------------------------------------------------------------------------------
# Frame identity for temporal synchronisation (docs/specs/2026-08-11-temporal-sync-design.md).
#
# tcn_shm_subscriber stamps every SHM frame with a nvidia::gxf::Timestamp; Holoscan surfaces it as
# the received message's acquisition timestamp. Python operators emit FRESH entities, so the stamp
# does not propagate by itself -- each stage must read it and pass it on via
# `op_output.emit(..., acq_timestamp=...)`. A stage that forgets loses identity SILENTLY, and the
# only symptom is that downstream grouping never matches.
# --------------------------------------------------------------------------------------------

NO_ACQ_TIMESTAMP = -1        # emit()'s sentinel for "no timestamp"


def acq_timestamp(op_input, port):
    """Acquisition timestamp of the message on `port`, or NO_ACQ_TIMESTAMP when absent.

    Returns -1 rather than None so the result can be passed straight to `emit(acq_timestamp=...)`.
    Note the accessor returns None (not 0) when no Timestamp is attached, so a genuinely absent
    stamp is distinguishable from a zero one.
    """
    try:
        t = op_input.get_acquisition_timestamp(port)
    except Exception:                     # accessor missing on older SDKs -- degrade, do not crash
        return NO_ACQ_TIMESTAMP
    return NO_ACQ_TIMESTAMP if t is None else int(t)


def acq_timestamp_consensus(op_input, port, log=None):
    """Acquisition timestamp for a multi-receiver port (IOSpec.ANY_SIZE), e.g. MaskCollectorOp.

    Every worker processes the SAME source frame, so all inbound messages should carry one
    timestamp. Disagreement means the workers have drifted onto different frames -- which would
    make any downstream frame-grouping quietly wrong -- so it is warned about rather than averaged
    away. The oldest is returned, matching the "publish the oldest complete group" rule the
    synchroniser uses.
    """
    try:
        stamps = op_input.get_acquisition_timestamps(port)
    except Exception:
        return NO_ACQ_TIMESTAMP
    vals = {int(t) for t in (stamps or []) if t is not None}
    if not vals:
        return NO_ACQ_TIMESTAMP
    if len(vals) > 1 and log is not None:
        log.warning(f"acq_timestamp mismatch across '{port}' receivers: {sorted(vals)} -- workers "
                    f"are on different source frames; downstream frame grouping will be wrong")
    return min(vals)


#: Entity components that a received tensor-map exposes as keys but which are NOT tensors.
#: Attaching an acquisition timestamp adds a `nvidia::gxf::Timestamp` component named "timestamp"
#: to the entity, and Holoscan's entity -> dict conversion enumerates ALL components, not just the
#: tensors. The extra key's value is None, so any consumer looping over `msg.keys()` and converting
#: blindly dies with a bewildering `ValueError: Unsupported dtype object`. Filter with
#: `tensor_names()` instead of iterating keys directly.
NON_TENSOR_COMPONENTS = frozenset({"timestamp"})


def tensor_names(msg):
    """Sorted tensor names in a received message, excluding non-tensor entity components.

    Use this anywhere the set of tensors is discovered from the message rather than known up front
    (a fixed `msg.get(camera_name)` needs no filtering).
    """
    if not msg:
        return []
    return sorted(k for k in msg.keys()
                  if k not in NON_TENSOR_COMPONENTS and msg.get(k) is not None)


