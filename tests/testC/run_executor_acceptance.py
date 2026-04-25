"""成员 C 六类基础动作真实验收脚本。

用途：
这个脚本不是 pytest 单元测试，而是给成员 C 在真实桌面环境中验证首版
executor 的人工验收工具。它会通过 `ExecutorWithRecorder` 执行六类基础
动作，并生成完整 run artifact：task.json、actions.jsonl、results.jsonl、
screenshot_log.jsonl、before/after screenshots 和 summary.json。

安全约定：
默认不执行任何真实 GUI 动作。必须显式传入 `--real-gui`，脚本才会移动
鼠标、点击、滚动、输入、发送快捷键。运行前请打开一个安全的测试窗口或
飞书测试环境，避免在生产聊天、文档或表单中误输入。

覆盖动作：
1. wait
2. click
3. double_click
4. scroll
5. type
6. hotkey

运行方式：
```powershell
/Users/lab340/miniconda3/envs/lark/bin/python tests\testC\run_executor_acceptance.py --real-gui --window-title 飞书
/Users/lab340/miniconda3/envs/lark/bin/python tests\testC\run_executor_acceptance.py --real-gui --fullscreen
/Users/lab340/miniconda3/envs/lark/bin/python tests\testC\run_executor_acceptance.py --real-gui --actions-json actions.json
```

自定义 actions JSON：
可以传入一个 JSON 数组，每项是 action_decision，例如：
```json
[
  {"action_type": "wait", "seconds": 0.5},
  {"action_type": "click", "x": 100, "y": 200},
  {"action_type": "type", "x": 120, "y": 220, "text": "CUA-Lark"}
]
```
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from executor import ExecutorWithRecorder  # noqa: E402


DEFAULT_TYPE_TEXT = "CUA-Lark executor acceptance"


def main() -> int:
    args = parse_args()
    if not args.real_gui:
        print("未传入 --real-gui，未执行任何真实桌面动作。")
        print("确认安全测试窗口已准备好后，再追加 --real-gui 运行。")
        return 0

    run_id = args.run_id or f"run_executor_acceptance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    actions = load_actions(args.actions_json) if args.actions_json else collect_interactive_actions(args)
    print("\n=== 成员 C 六类基础动作真实验收 ===")
    print(f"run_id: {run_id}")
    print(f"artifact_root: {args.artifact_root}")
    print(f"window_title: {None if args.fullscreen else args.window_title}")
    print(f"动作数: {len(actions)}")
    input("确认即将执行真实鼠标键盘动作；按 Enter 开始，Ctrl+C 取消。")

    runner = ExecutorWithRecorder(
        run_id=run_id,
        task={
            "name": "executor_acceptance",
            "actions": actions,
            "manual": True,
        },
        artifact_root=args.artifact_root,
        window_title=None if args.fullscreen else args.window_title,
        monitor_index=args.monitor_index,
        focus_window=not args.no_focus_window,
        focus_delay_seconds=args.focus_delay_seconds,
    )

    all_ok = True
    for step_idx, action in enumerate(actions, start=1):
        print(f"\n--- Step {step_idx}: {json.dumps(action, ensure_ascii=False)}")
        result = runner.run_step(action, step_idx=step_idx)
        result_payload = result.action_result.to_dict()
        all_ok = all_ok and result.action_result.ok
        print(json.dumps(result_payload, ensure_ascii=False, indent=2))

    summary = runner.finalize(status="success" if all_ok else "failed")
    print("\n=== 验收输出 ===")
    print(f"summary: {summary['artifact_paths']['summary_path']}")
    print(f"screenshots: {summary['artifact_paths']['screenshots_dir']}")
    print(f"status: {summary['status']}")
    return 0 if all_ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="成员 C executor 真实 GUI 验收脚本")
    parser.add_argument("--real-gui", action="store_true", help="确认执行真实鼠标键盘动作")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/runs")
    parser.add_argument("--window-title", default="飞书")
    parser.add_argument("--fullscreen", action="store_true", help="使用全屏截图而不是窗口截图")
    parser.add_argument("--monitor-index", type=int, default=1)
    parser.add_argument("--no-focus-window", action="store_true")
    parser.add_argument("--focus-delay-seconds", type=float, default=0.5)
    parser.add_argument("--actions-json", default=None, help="读取自定义 action_decision JSON 数组")
    parser.add_argument("--type-text", default=DEFAULT_TYPE_TEXT)
    return parser.parse_args()


def load_actions(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("actions-json must contain a JSON array")
    actions: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each action item must be an object")
        actions.append(dict(item))
    return actions


def collect_interactive_actions(args: argparse.Namespace) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [{"action_type": "wait", "seconds": 0.5}]
    print("\n将依次采集 click、double_click、scroll、type 的鼠标位置。")
    click_point = prompt_cursor_point("把鼠标移动到一个安全可点击位置，然后按 Enter")
    actions.append({"action_type": "click", "x": click_point[0], "y": click_point[1]})

    double_click_point = prompt_cursor_point("把鼠标移动到一个安全可双击位置，然后按 Enter")
    actions.append(
        {
            "action_type": "double_click",
            "x": double_click_point[0],
            "y": double_click_point[1],
        }
    )

    scroll_point = prompt_cursor_point("把鼠标移动到一个安全可滚动区域，然后按 Enter")
    actions.append(
        {
            "action_type": "scroll",
            "x": scroll_point[0],
            "y": scroll_point[1],
            "direction": "down",
        }
    )

    type_point = prompt_cursor_point("把鼠标移动到安全输入框，然后按 Enter")
    actions.append(
        {
            "action_type": "type",
            "x": type_point[0],
            "y": type_point[1],
            "text": args.type_text,
        }
    )
    actions.append({"action_type": "hotkey", "keys": ["esc"]})
    return actions


def prompt_cursor_point(message: str) -> tuple[int, int]:
    input(message)
    try:
        import pyautogui
    except ImportError as exc:
        raise RuntimeError("pyautogui is required to collect cursor position") from exc
    position = pyautogui.position()
    return int(position.x), int(position.y)


if __name__ == "__main__":
    raise SystemExit(main())
