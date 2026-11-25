"""Phase-SHA V10「衔尾蛇-64」R2 只读诊断与反例验证套件。

严格隔离约束：
  - 仅在 v10_ouroboros64/diagnostics/ 下创建诊断脚本与输出 JSON；
  - 严禁修改 ouroboros_core/, ouroboros_net/, ouroboros_mcts/, ouroboros_train/, ouroboros_bench/ 及 runs/。
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> Path:
    """确保工作区根目录位于 sys.path 中，使脚本既支持 `-m` 又支持直接路径执行。"""
    repo_root = Path(__file__).resolve().parents[2]
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return repo_root
