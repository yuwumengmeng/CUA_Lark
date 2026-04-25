"""pytest 测试导入路径配置脚本。

这个脚本把仓库根目录加入 `sys.path`，确保测试可以直接导入
`planner`、`capture` 等项目内包。
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
