"""Member C IM GUI skill tests.

The skills are tested with a fake backend, so no real Feishu window is touched.
"""

import unittest

from executor import (
    IMExecutionSkills,
    click_candidate,
    focus_and_type,
    open_search_by_hotkey,
    press_enter_to_send,
    scroll_result_list,
    wait_for_page_change,
)
from schemas import SOURCE_VLM_VISUAL


class FakeDesktopBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def click(self, x: int, y: int) -> None:
        self.calls.append(("click", (x, y)))

    def double_click(self, x: int, y: int) -> None:
        self.calls.append(("double_click", (x, y)))

    def scroll(self, amount: int, x: int | None = None, y: int | None = None) -> None:
        self.calls.append(("scroll", (amount, x, y)))

    def type_text(self, text: str, *, interval: float = 0.0) -> None:
        self.calls.append(("type_text", (text, interval)))

    def hotkey(self, keys: list[str]) -> None:
        self.calls.append(("hotkey", tuple(keys)))

    def wait(self, seconds: float) -> None:
        self.calls.append(("wait", seconds))


class CandidateObject:
    def to_dict(self) -> dict[str, object]:
        return candidate("elem_obj", [50, 60])


class IMExecutionSkillsTests(unittest.TestCase):
    def test_click_candidate_uses_candidate_center(self):
        backend = FakeDesktopBackend()
        skills = IMExecutionSkills.create(backend=backend)

        result = skills.click_candidate(candidate("elem_chat", [100, 200]))

        self.assertTrue(result.ok)
        self.assertEqual(result.target_candidate_id, "elem_chat")
        self.assertEqual(result.planned_click_point, [100, 200])
        self.assertEqual(result.actual_click_point, [100, 200])
        self.assertEqual(backend.calls, [("wait", 0.08), ("click", (100, 200))])

    def test_focus_and_type_uses_gui_click_then_text_input(self):
        backend = FakeDesktopBackend()
        skills = IMExecutionSkills.create(backend=backend)

        result = skills.focus_and_type("Hello CUA-Lark", candidate=CandidateObject())

        self.assertTrue(result.ok)
        self.assertEqual(result.target_candidate_id, "elem_obj")
        self.assertEqual(
            backend.calls,
            [
                ("click", (50, 60)),
                ("wait", 0.08),
                ("type_text", ("Hello CUA-Lark", 0.0)),
            ],
        )

    def test_click_candidate_accepts_vlm_selected_candidate(self):
        backend = FakeDesktopBackend()
        skills = IMExecutionSkills.create(backend=backend)
        vlm_candidate = candidate("vlm_elem_chat", [77, 88])
        vlm_candidate["source"] = SOURCE_VLM_VISUAL
        vlm_candidate["confidence"] = 0.82

        result = skills.click_candidate(
            vlm_candidate,
            step_name="click_vlm_candidate",
            reason="click VLM recommended candidate",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.target_candidate_id, "vlm_elem_chat")
        self.assertEqual(result.planned_click_point, [77, 88])
        self.assertEqual(result.actual_click_point, [77, 88])
        self.assertEqual(backend.calls, [("wait", 0.08), ("click", (77, 88))])

    def test_enter_search_wait_and_scroll_skills(self):
        backend = FakeDesktopBackend()
        skills = IMExecutionSkills.create(backend=backend)

        search_result = skills.open_search_by_hotkey()
        enter_result = skills.press_enter_to_send()
        wait_result = skills.wait_for_page_change(seconds=0.5)
        scroll_result = skills.scroll_result_list(direction="down", amount=3, x=10, y=20)

        self.assertTrue(search_result.ok)
        self.assertTrue(enter_result.ok)
        self.assertTrue(wait_result.ok)
        self.assertTrue(scroll_result.ok)
        self.assertEqual(
            backend.calls,
            [
                ("hotkey", ("ctrl", "k")),
                ("hotkey", ("enter",)),
                ("wait", 0.5),
                ("scroll", (3, 10, 20)),
            ],
        )

    def test_module_level_helpers_create_skills(self):
        click_backend = FakeDesktopBackend()
        type_backend = FakeDesktopBackend()
        search_backend = FakeDesktopBackend()
        enter_backend = FakeDesktopBackend()
        wait_backend = FakeDesktopBackend()
        scroll_backend = FakeDesktopBackend()

        self.assertTrue(click_candidate(candidate("elem", [1, 2]), backend=click_backend).ok)
        self.assertTrue(
            focus_and_type("hi", x=3, y=4, backend=type_backend).ok
        )
        self.assertTrue(open_search_by_hotkey(backend=search_backend).ok)
        self.assertTrue(press_enter_to_send(backend=enter_backend).ok)
        self.assertTrue(wait_for_page_change(seconds=0.1, backend=wait_backend).ok)
        self.assertTrue(scroll_result_list(amount=-2, backend=scroll_backend).ok)

        self.assertEqual(click_backend.calls[-1], ("click", (1, 2)))
        self.assertEqual(type_backend.calls[0], ("click", (3, 4)))
        self.assertEqual(search_backend.calls, [("hotkey", ("ctrl", "k"))])
        self.assertEqual(enter_backend.calls, [("hotkey", ("enter",))])
        self.assertEqual(wait_backend.calls, [("wait", 0.1)])
        self.assertEqual(scroll_backend.calls, [("scroll", (-2, None, None))])


def candidate(candidate_id: str, center: list[int]) -> dict[str, object]:
    x, y = center
    return {
        "id": candidate_id,
        "text": "target",
        "bbox": [x - 5, y - 5, x + 5, y + 5],
        "center": center,
        "role": "Button",
        "clickable": True,
        "editable": False,
        "source": "uia",
        "confidence": 0.9,
    }


if __name__ == "__main__":
    unittest.main()
