"""
Phase-SHA V10「衔尾蛇-64」M3 第一梯队零重训突破原型包 (`v10_ouroboros64.prototypes`)。
"""

from .p1_clean_cdcl import (
    CleanDynamicNeuralCDCLSolver,
    FullCrystalVarMap,
    _build_full_crystal_var_map,
)
from .p2_adaptive_probe_xor import (
    SyncedDynamicXORSolver,
    ZeroWasteBoundaryProber,
)
from .p3_dual_bridge_router import (
    BridgeableSHA256XORCNFEncoder,
    BridgeExtractionResult,
    CarryLockHybridRouter,
    DualPlaneEquivalenceBridge,
    SafeCMSSubprocessRunner,
    SignedUnionFind,
)
from .p4_local_wormhole_mcts import (
    LocalWindowWormholeMCTS,
    compute_window_theory_var_count,
    local_window_gf2_propagate,
)

__all__ = [
    "CleanDynamicNeuralCDCLSolver",
    "FullCrystalVarMap",
    "_build_full_crystal_var_map",
    "ZeroWasteBoundaryProber",
    "SyncedDynamicXORSolver",
    "BridgeableSHA256XORCNFEncoder",
    "SignedUnionFind",
    "BridgeExtractionResult",
    "DualPlaneEquivalenceBridge",
    "SafeCMSSubprocessRunner",
    "CarryLockHybridRouter",
    "local_window_gf2_propagate",
    "compute_window_theory_var_count",
    "LocalWindowWormholeMCTS",
]
