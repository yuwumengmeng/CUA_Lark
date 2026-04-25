"""成员 A 4.23 初版 UIA / Accessibility 信息抽取测试。

测试目标：
本文件验证 `perception/uia.py` 的 UIA 输出协议是否稳定，包括：
1. `extract_uia_elements` 能通过 fake backend 返回 `UIAElement`。
2. UIA 输出字段固定为 `name / role / bbox / enabled / visible`。
3. fake control tree 能转换为统一 UIA 元素列表。
4. invisible、非法 bbox、非法配置会被正确过滤或拒绝。

重要约定：
这里不访问真实 Windows UIA 树，不依赖当前电脑是否打开飞书。测试使用
fake backend 和 fake control，只验证接口协议和数据转换逻辑。

直接运行：
```powershell
python tests\testA\test_uia.py
```

通过 pytest 运行：
```powershell
python -m pytest tests\testA\test_uia.py
```

真实 UIA 手动验证请先打开飞书，然后运行：
```powershell
python -c "from perception.uia import UIAConfig, extract_uia_elements; print([e.to_dict() for e in extract_uia_elements(config=UIAConfig(window_title='飞书'))[:20]])"
```
"""

import sys
import unittest
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

uia_module = import_module("perception.uia")
UIAConfig = uia_module.UIAConfig
UIAElement = uia_module.UIAElement
UIAError = uia_module.UIAError
extract_from_control_tree = uia_module.extract_from_control_tree
extract_uia_elements = uia_module.extract_uia_elements


class FakeRect:
    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


class FakeControl:
    def __init__(
        self,
        *,
        name: str,
        role: str,
        rect: FakeRect | tuple[int, int, int, int] | None,
        enabled: bool = True,
        offscreen: bool = False,
        children: list["FakeControl"] | None = None,
    ) -> None:
        self.Name = name
        self.ControlTypeName = role
        self.BoundingRectangle = rect
        self.IsEnabled = enabled
        self.IsOffscreen = offscreen
        self._children = children or []

    def GetChildren(self) -> list["FakeControl"]:
        return self._children


class FakeUIABackend:
    def __init__(self) -> None:
        self.seen_windows: list[object] = []
        self.seen_configs: list[UIAConfig] = []

    def extract(self, window: object, config: UIAConfig) -> list[UIAElement]:
        self.seen_windows.append(window)
        self.seen_configs.append(config)
        return [
            UIAElement(
                name="搜索",
                role="EditControl",
                bbox=[10, 20, 160, 48],
                enabled=True,
                visible=True,
            )
        ]


class UIATests(unittest.TestCase):
    def test_extract_uia_elements_uses_backend_and_stable_fields(self):
        backend = FakeUIABackend()
        config = UIAConfig(window_title="飞书", max_depth=3)

        elements = extract_uia_elements("飞书", config=config, backend=backend)

        self.assertEqual(len(elements), 1)
        self.assertEqual(backend.seen_windows, ["飞书"])
        self.assertEqual(backend.seen_configs, [config])
        self.assertEqual(
            elements[0].to_dict(),
            {
                "version": "uia_element.v1",
                "name": "搜索",
                "role": "EditControl",
                "bbox": [10, 20, 160, 48],
                "enabled": True,
                "visible": True,
            },
        )

    def test_control_tree_is_converted_to_uia_elements(self):
        root = FakeControl(
            name="飞书",
            role="WindowControl",
            rect=FakeRect(0, 0, 1000, 800),
            children=[
                FakeControl(
                    name="消息",
                    role="ButtonControl",
                    rect=FakeRect(20, 80, 80, 120),
                ),
                FakeControl(
                    name="搜索",
                    role="EditControl",
                    rect=(100, 20, 320, 52),
                ),
            ],
        )

        elements = extract_from_control_tree(root, UIAConfig(max_depth=2))

        self.assertEqual([element.name for element in elements], ["飞书", "消息", "搜索"])
        self.assertEqual(elements[0].bbox, [0, 0, 1000, 800])
        self.assertEqual(elements[2].role, "EditControl")

    def test_invisible_and_invalid_controls_are_filtered_by_default(self):
        root = FakeControl(
            name="飞书",
            role="WindowControl",
            rect=FakeRect(0, 0, 1000, 800),
            children=[
                FakeControl(
                    name="",
                    role="GroupControl",
                    rect=FakeRect(1, 1, 999, 799),
                ),
                FakeControl(
                    name="",
                    role="ButtonControl",
                    rect=FakeRect(100, 100, 140, 140),
                ),
                FakeControl(
                    name="隐藏按钮",
                    role="ButtonControl",
                    rect=FakeRect(20, 80, 80, 120),
                    offscreen=True,
                ),
                FakeControl(
                    name="非法区域",
                    role="ButtonControl",
                    rect=FakeRect(10, 10, 10, 20),
                ),
                FakeControl(
                    name="无区域",
                    role="ButtonControl",
                    rect=None,
                ),
            ],
        )

        visible_only = extract_from_control_tree(root, UIAConfig(max_depth=2))
        with_invisible = extract_from_control_tree(
            root,
            UIAConfig(max_depth=2, include_invisible=True),
        )

        self.assertEqual([element.role for element in visible_only], ["WindowControl", "ButtonControl"])
        self.assertEqual(
            [element.name for element in with_invisible],
            ["飞书", "", "隐藏按钮"],
        )
        self.assertFalse(with_invisible[2].visible)

    def test_max_depth_limits_traversal(self):
        nested = FakeControl(
            name="root",
            role="WindowControl",
            rect=FakeRect(0, 0, 100, 100),
            children=[
                FakeControl(
                    name="level_1",
                    role="PaneControl",
                    rect=FakeRect(1, 1, 90, 90),
                    children=[
                        FakeControl(
                            name="level_2",
                            role="ButtonControl",
                            rect=FakeRect(2, 2, 80, 80),
                        )
                    ],
                )
            ],
        )

        elements = extract_from_control_tree(nested, UIAConfig(max_depth=1))

        self.assertEqual([element.name for element in elements], ["root", "level_1"])

    def test_invalid_uia_element_and_config_are_rejected(self):
        with self.assertRaises(UIAError):
            UIAElement(name="搜索", role="EditControl", bbox=[10, 10, 5, 20], enabled=True, visible=True)

        with self.assertRaises(UIAError):
            UIAElement(name="搜索", role="EditControl", bbox=[0, 0, 10, 10], enabled="yes", visible=True)

        with self.assertRaises(UIAError):
            UIAConfig(window_title="")

        with self.assertRaises(UIAError):
            UIAConfig(max_depth=-1)

        with self.assertRaises(UIAError):
            UIAConfig(max_elements=0)


if __name__ == "__main__":
    unittest.main()
