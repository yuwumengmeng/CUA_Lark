# ABC 联合验收脚本

`run_joint_abc_acceptance.py` 是第一周期 A/B/C 联合人工验收脚本，不是 pytest。

脚本按场景运行真实链路：

```text
A get_ui_state() -> B MiniPlanner -> C ExecutorWithRecorder
```

## 当前验收范围

第一周期联合验收保留 4 个核心单步场景：

1. 点击侧边栏模块的云文档按钮
2. 搜索框输入关键词
3. 点击群聊
4. 打开文档入口

“返回上一级”暂不作为第一周期 ABC 联合验收场景，后续可单独放到导航/恢复类用例。

## 推荐运行

```powershell
python tests\testABC\run_joint_abc_acceptance.py --real-gui --yes --window-title 飞书 --chat-name 测试群 --search-text 项目周报
```

运行方式：

- 每个场景开始前，脚本会停下来让你手动切换飞书界面。
- 准备好后按一次 Enter。
- 场景内动作会自动执行，不再确认坐标。
- 输出会写入 `artifacts/runs/<run_id>/`。
- 默认使用快速模式：只在动作前跑 A 的 OCR/UIA 感知，动作后由 C 保留截图和日志。
- 如果要做更严格但更慢的业务状态校验，追加 `--strict-after-checks`。

主要输出：

```text
abc_acceptance_summary.json
abc_acceptance_report.md
actions.jsonl
results.jsonl
screenshot_log.jsonl
screenshots/
ui_state_step_xxx_before.json
ui_state_step_xxx_after.json
```

## 当前观察

- 动作执行本身不慢；本轮 C 的 click/type 执行一般在 100-250ms。
- 体感慢主要来自每步多次截图、窗口激活等待、OCR、UIA 抽取和 after 状态校验，不是 JSON 数据传输量大。
- 群聊场景曾存在误命中风险：OCR 可能把“测试群”拆成“测试”和单字“群”。当前 B 已禁止把单字候选当成多字目标的弱匹配，并会优先选择“测试”这类有意义的部分文本。
- 后续优化方向是继续让 B 优先选择完整文本、可点击行候选和聊天列表区域候选。
