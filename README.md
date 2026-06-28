# Graphwar 点位检测

用简单的 OpenCV 颜色分割和轮廓筛选，从截图中定位黄色玩家与蓝色目标，并把像素中心映射为游戏坐标。两类点统一输出，不在结果中区分。

## 图形界面

```powershell
uv sync
uv run python ui.py
```

操作流程：

1. 点击“截取当前屏幕”或“打开截图”。
2. 在坐标区域中按顺序点击轨迹点；靠近玩家或目标时会自动吸附。
3. 调整算法参数，点击“生成预览”检查表达式。
4. 点击“确定并复制”，表达式会写入系统剪贴板。

绿色选点表示已吸附，红色选点表示自由坐标。右键画布可撤销最后一个点。

## 命令行检测

```powershell
uv sync
uv run python main.py screenshot.png
```

保存可视化检测结果：

```powershell
uv run python main.py screenshot.png --debug-output detected.png
```

程序通常会自动寻找白色坐标区域。若截图主题变化导致自动识别失败，可手工传入区域内边界：

```powershell
uv run python main.py screenshot.png --board LEFT TOP RIGHT BOTTOM
```

输出是只含 `x`、`y` 游戏坐标的 JSON 数组，不区分玩家和目标。图像坐标的 y 轴向下，转换后的游戏 y 轴向上。
