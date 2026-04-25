"""成员 C ActionResult 协议测试。

测试目标：
本文件验证 `schemas/action_result.py` 的执行结果协议是否稳定，包括：
1. 成功/失败结果都输出 `ok / action_type / start_ts / end_ts / error`。
2. `duration_ms` 能从时间戳自动计算，供 recorder 统计。
3. 非法布尔值、时间顺序和错误字段会被拒绝。

重要约定：
这里不访问真实桌面、不执行鼠标键盘动作，只验证 schema 层。

运行方式：
```powershell
python -m pytest tests\testC\test_action_result.py
python tests\testC\test_action_result.py
```
"""

import unittest

from schemas import ActionResult, ActionResultSchemaError, action_result_from_exception


class ActionResultTests(unittest.TestCase):
    def test_success_result_outputs_stable_fields(self):
        result = ActionResult.success(
            action_type="click",
            start_ts="2026-04-24T10:00:00.000Z",
            end_ts="2026-04-24T10:00:00.125Z",
        )

        self.assertEqual(
            result.to_dict(),
            {
                "version": "action_result.v1",
                "ok": True,
                "action_type": "click",
                "start_ts": "2026-04-24T10:00:00.000Z",
                "end_ts": "2026-04-24T10:00:00.125Z",
                "error": None,
                "duration_ms": 125,
            },
        )

    def test_failure_result_roundtrip(self):
        result = action_result_from_exception(
            action_type="drag",
            start_ts="2026-04-24T10:00:00.000Z",
            end_ts="2026-04-24T10:00:00.050Z",
            error=ValueError("unsupported action_type: drag"),
        )
        restored = ActionResult.from_dict(result.to_dict())

        self.assertFalse(restored.ok)
        self.assertEqual(restored.action_type, "drag")
        self.assertEqual(restored.error, "unsupported action_type: drag")
        self.assertEqual(restored.duration_ms, 50)

    def test_invalid_result_data_is_rejected(self):
        with self.assertRaises(ActionResultSchemaError):
            ActionResult(
                ok="yes",
                action_type="click",
                start_ts="2026-04-24T10:00:00Z",
                end_ts="2026-04-24T10:00:01Z",
            )

        with self.assertRaises(ActionResultSchemaError):
            ActionResult(
                ok=True,
                action_type="click",
                start_ts="2026-04-24T10:00:01Z",
                end_ts="2026-04-24T10:00:00Z",
            )

        with self.assertRaises(ActionResultSchemaError):
            ActionResult(
                ok=False,
                action_type="click",
                start_ts="2026-04-24T10:00:00Z",
                end_ts="2026-04-24T10:00:01Z",
            )

        with self.assertRaises(ActionResultSchemaError):
            ActionResult(
                ok=True,
                action_type="click",
                start_ts="2026-04-24T10:00:00Z",
                end_ts="2026-04-24T10:00:01Z",
                error="unexpected",
            )


if __name__ == "__main__":
    unittest.main()
