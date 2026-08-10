"""TcnDatasetReplayerOp: deterministic dataset-replay stand-in for TcnShmSubscriberOp.

Plain Python package (no CMakeLists), imported as
``from operators.tcn_artekmed.tcn_dataset_replayer import TcnDatasetReplayerOp`` --
mirrors ``operators/tcn_artekmed/tcn_util``.
"""
from .dataset_replayer_op import TcnDatasetReplayerOp

__all__ = ["TcnDatasetReplayerOp"]
