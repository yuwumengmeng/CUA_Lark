"""成员 A 4.23 初版 UIA / Accessibility 信息抽取。

职责边界：
本文件只负责从 Windows UI Automation 树中抽取基础可交互元素信息，并
输出统一的 UIA 元素结构。它不负责截图采集、不负责 OCR、不负责 OCR+UIA
候选合并，也不负责最终 clickable/editable 规则。后续 candidate builder
应该消费这里的 `UIAElement`，再转换成统一 `UICandidate`。

公开接口：
`extract_uia_elements(window=None, config=None, backend=None)`
    首版 UIA 入口。`window` 可以是 UIA control 对象、窗口标题字符串或
    `None`。输出 `list[UIAElement]`。
`UIAElement.to_dict()`
    返回稳定 dict，字段为 `name / role / bbox / enabled / visible`。
`UIAutomationBackend`
    默认真实 UIA 后端。使用 Python 包 `uiautomation` 读取 Windows UIA 树。

输出结构：
`name`
    UIA 元素名称或标题。允许为空字符串，但默认只保留无 name 的
    Button/Edit/Text/Document 类元素。
`role`
    UIA control type，例如 ButtonControl、EditControl、TextControl。
`bbox`
    元素在屏幕坐标系中的位置，格式为 `[x1, y1, x2, y2]`。
`enabled`
    元素是否可用。
`visible`
    元素是否未被 UIA 标记为 offscreen。

真实 UIA 环境要求：
1. 当前系统为 Windows。
2. Python 依赖需要安装 `uiautomation`。
3. 飞书窗口需要处于可访问状态；Electron 应用可能只能暴露部分控件。

使用示例：
```python
from perception.uia import UIAConfig, extract_uia_elements

elements = extract_uia_elements(config=UIAConfig(window_title="飞书", max_depth=5))
for element in elements:
    print(element.to_dict())
```

协作规则：
1. UIA 输出字段名固定为 `name / role / bbox / enabled / visible`。
2. `bbox` 沿用屏幕坐标系，和首版全屏截图坐标保持一致。
3. 默认过滤无 name 的 GroupControl 等结构容器，减少 Electron 重复噪声。
4. 单元测试使用 fake control tree，不依赖真实飞书窗口；真实 UIA 验证应
   用手动命令或可视化检查脚本完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


UIA_SCHEMA_VERSION = "uia_element.v1"


class UIAError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class UIAElement:
    name: str
    role: str
    bbox: list[int]
    enabled: bool
    visible: bool
    version: str = UIA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        role = str(self.role).strip() or "unknown"
        bbox = normalize_bbox(self.bbox)
        if not isinstance(self.enabled, bool):
            raise UIAError("UIAElement enabled must be a boolean")
        if not isinstance(self.visible, bool):
            raise UIAError("UIAElement visible must be a boolean")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "bbox", bbox)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "role": self.role,
            "bbox": self.bbox,
            "enabled": self.enabled,
            "visible": self.visible,
        }


@dataclass(frozen=True, slots=True)
class UIAConfig:
    window_title: str | None = None
    max_depth: int = 6
    max_elements: int = 500
    include_invisible: bool = False

    def __post_init__(self) -> None:
        if self.window_title is not None and not self.window_title.strip():
            raise UIAError("UIAConfig window_title cannot be empty")
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int) or self.max_depth < 0:
            raise UIAError("UIAConfig max_depth must be a non-negative integer")
        if isinstance(self.max_elements, bool) or not isinstance(self.max_elements, int) or self.max_elements <= 0:
            raise UIAError("UIAConfig max_elements must be a positive integer")
        if not isinstance(self.include_invisible, bool):
            raise UIAError("UIAConfig include_invisible must be a boolean")


class UIABackend(Protocol):
    def extract(self, window: Any, config: UIAConfig) -> list[UIAElement]:
        ...


class UIAutomationBackend:
    def extract(self, window: Any, config: UIAConfig) -> list[UIAElement]:
        try:
            import uiautomation as auto
        except ImportError as exc:
            raise UIAError(
                "uiautomation is not installed. Install dependencies from requirements.txt."
            ) from exc

        root = self._resolve_root(window, config, auto)
        return extract_from_control_tree(root, config)

    def _resolve_root(self, window: Any, config: UIAConfig, auto_module: Any) -> Any:
        if window is not None and not isinstance(window, str):
            return window

        title = window if isinstance(window, str) else config.window_title
        if title:
            matched = _find_top_level_window_by_title(auto_module.GetRootControl(), title)
            if matched is None:
                raise UIAError(f"UIA window was not found by title: {title}")
            return matched

        focused = _safe_call(auto_module, "GetFocusedControl")
        if focused is not None:
            window_control = _top_level_window_from_control(focused)
            if window_control is not None:
                return window_control
            return focused

        return auto_module.GetRootControl()


def extract_uia_elements(
    window: Any = None,
    *,
    config: UIAConfig | None = None,
    backend: UIABackend | None = None,
) -> list[UIAElement]:
    uia_config = config or UIAConfig()
    uia_backend = backend or UIAutomationBackend()
    return uia_backend.extract(window, uia_config)


def extract_from_control_tree(root: Any, config: UIAConfig | None = None) -> list[UIAElement]:
    uia_config = config or UIAConfig()
    elements: list[UIAElement] = []
    stack: list[tuple[Any, int]] = [(root, 0)]

    while stack and len(elements) < uia_config.max_elements:
        control, depth = stack.pop()
        element = element_from_control(control)
        if element is not None and (element.visible or uia_config.include_invisible):
            elements.append(element)

        if depth >= uia_config.max_depth:
            continue

        children = _children(control)
        for child in reversed(children):
            stack.append((child, depth + 1))

    return elements


def element_from_control(control: Any) -> UIAElement | None:
    bbox = _control_bbox(control)
    if bbox is None:
        return None

    name = _string_attr(control, "Name")
    role = _string_attr(control, "ControlTypeName") or _string_attr(control, "ClassName") or "unknown"
    enabled = _bool_attr(control, "IsEnabled", default=True)
    offscreen = _bool_attr(control, "IsOffscreen", default=False)
    visible = not offscreen

    if not name and role == "unknown":
        return None
    if not _should_keep_element(name, role):
        return None

    return UIAElement(
        name=name,
        role=role,
        bbox=bbox,
        enabled=enabled,
        visible=visible,
    )


def normalize_bbox(bbox: list[int]) -> list[int]:
    if len(bbox) != 4:
        raise UIAError("UIAElement bbox must contain four integers")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in bbox):
        raise UIAError("UIAElement bbox values must be integers")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise UIAError("UIAElement bbox must satisfy x2 > x1 and y2 > y1")
    return [x1, y1, x2, y2]


def _control_bbox(control: Any) -> list[int] | None:
    rect = getattr(control, "BoundingRectangle", None)
    if rect is None:
        return None

    values = _rect_values(rect)
    if values is None:
        return None

    x1, y1, x2, y2 = values
    try:
        return normalize_bbox([x1, y1, x2, y2])
    except UIAError:
        return None


def _rect_values(rect: Any) -> tuple[int, int, int, int] | None:
    attr_names = (("left", "top", "right", "bottom"), ("Left", "Top", "Right", "Bottom"))
    for names in attr_names:
        if all(hasattr(rect, name) for name in names):
            return tuple(int(getattr(rect, name)) for name in names)  # type: ignore[return-value]

    if isinstance(rect, (list, tuple)) and len(rect) == 4:
        return tuple(int(value) for value in rect)  # type: ignore[return-value]

    return None


def _children(control: Any) -> list[Any]:
    try:
        children = control.GetChildren()
    except Exception:
        return []
    if children is None:
        return []
    return list(children)


def _find_top_level_window_by_title(root: Any, title: str) -> Any | None:
    title_text = title.strip()
    for child in _children(root):
        name = _string_attr(child, "Name")
        if title_text in name:
            return child
    return None


def _top_level_window_from_control(control: Any) -> Any | None:
    current = control
    last_window = None
    for _ in range(20):
        role = _string_attr(current, "ControlTypeName")
        if "Window" in role:
            last_window = current
        try:
            parent = current.GetParentControl()
        except Exception:
            break
        if parent is None:
            break
        current = parent
    return last_window


def _safe_call(module: Any, method_name: str) -> Any | None:
    method = getattr(module, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


def _string_attr(obj: Any, attr_name: str) -> str:
    value = getattr(obj, attr_name, "")
    if value is None:
        return ""
    return str(value).strip()


def _bool_attr(obj: Any, attr_name: str, *, default: bool) -> bool:
    value = getattr(obj, attr_name, default)
    if isinstance(value, bool):
        return value
    return default


def _should_keep_element(name: str, role: str) -> bool:
    if name:
        return True
    keep_role_keywords = ("Button", "Edit", "Text", "Document")
    return any(keyword in role for keyword in keep_role_keywords)
