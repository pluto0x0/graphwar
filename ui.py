from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageGrab, ImageTk

from algo import expression_text
from main import Detection, detect_points


class GraphwarUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Graphwar 轨迹生成器")
        self.root.geometry("1180x780")
        self.root.minsize(900, 620)

        self.image: np.ndarray | None = None
        self.board: tuple[int, int, int, int] | None = None
        self.detected: list[Detection] = []
        self.selected: list[tuple[float, float, bool]] = []
        self.photo: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.display_origin = (0.0, 0.0)

        self.factor_var = tk.StringVar(value="20")
        self.limit_var = tk.StringVar(value="10")
        self.freq_var = tk.StringVar(value="10")
        self.width_var = tk.StringVar(value="0.5")
        self.snap_var = tk.StringVar(value="32")
        self.status_var = tk.StringVar(value="请截取屏幕或打开截图")

        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="截取当前屏幕", command=self.capture_screen).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="打开截图…", command=self.open_image).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.LEFT, padx=16)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        canvas_frame = ttk.Frame(body)
        self.canvas = tk.Canvas(
            canvas_frame,
            background="#202020",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", lambda _event: self.undo_point())
        self.canvas.bind("<Configure>", lambda _event: self.render())
        body.add(canvas_frame, weight=4)

        panel = ttk.Frame(body, padding=(12, 4))
        body.add(panel, weight=1)

        ttk.Label(panel, text="已选点（按点击顺序）").pack(anchor=tk.W)
        self.point_list = tk.Listbox(panel, height=11, activestyle="none")
        self.point_list.pack(fill=tk.X, pady=(4, 6))

        point_buttons = ttk.Frame(panel)
        point_buttons.pack(fill=tk.X)
        ttk.Button(point_buttons, text="撤销末点", command=self.undo_point).pack(side=tk.LEFT)
        ttk.Button(point_buttons, text="清空", command=self.clear_points).pack(side=tk.LEFT, padx=6)

        ttk.Separator(panel).pack(fill=tk.X, pady=14)
        params = ttk.LabelFrame(panel, text="算法参数", padding=10)
        params.pack(fill=tk.X)
        self._param_row(params, 0, "factor", self.factor_var)
        self._param_row(params, 1, "limit（留空关闭）", self.limit_var)
        self._param_row(params, 2, "freq", self.freq_var)
        self._param_row(params, 3, "width", self.width_var)
        self._param_row(params, 4, "吸附半径/像素", self.snap_var)

        actions = ttk.Frame(panel)
        actions.pack(fill=tk.X, pady=12)
        ttk.Button(actions, text="生成预览", command=self.generate_preview).pack(side=tk.LEFT)
        ttk.Button(actions, text="确定并复制", command=self.copy_expression).pack(
            side=tk.LEFT, padx=8
        )

        ttk.Label(panel, text="表达式").pack(anchor=tk.W)
        self.expression = tk.Text(panel, height=9, wrap=tk.WORD)
        self.expression.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    @staticmethod
    def _param_row(parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(parent, textvariable=variable, width=11).grid(
            row=row, column=1, sticky=tk.E, padx=(8, 0), pady=3
        )
        parent.columnconfigure(0, weight=1)

    def capture_screen(self) -> None:
        self.root.withdraw()
        self.root.after(300, self._finish_capture)

    def _finish_capture(self) -> None:
        try:
            screenshot = ImageGrab.grab(all_screens=True).convert("RGB")
            image = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            self.root.deiconify()
            messagebox.showerror("截屏失败", str(exc), parent=self.root)
            return
        self.root.deiconify()
        self.root.lift()
        self.load_image(image)

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="选择截图",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")],
        )
        if not filename:
            return
        try:
            rgb = np.asarray(Image.open(Path(filename)).convert("RGB"))
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc), parent=self.root)
            return
        self.load_image(image)

    def load_image(self, image: np.ndarray) -> None:
        try:
            board, detected = detect_points(image)
        except ValueError as exc:
            messagebox.showerror("识别失败", str(exc), parent=self.root)
            return
        self.image = image
        self.board = board
        self.detected = detected
        self.selected.clear()
        self._refresh_point_list()
        self.status_var.set(f"识别到 {len(detected)} 个玩家/目标；左键选点，右键撤销")
        self.render()

    def render(self) -> None:
        self.canvas.delete("all")
        if self.image is None or self.board is None:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text="截取当前屏幕或打开已有截图",
                fill="#dddddd",
                font=("TkDefaultFont", 14),
            )
            return

        left, top, right, bottom = self.board
        crop = self.image[top : bottom + 1, left : right + 1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        self.display_scale = min(canvas_w / image.width, canvas_h / image.height)
        draw_w = max(1, round(image.width * self.display_scale))
        draw_h = max(1, round(image.height * self.display_scale))
        self.display_origin = ((canvas_w - draw_w) / 2, (canvas_h - draw_h) / 2)
        resized = image.resize((draw_w, draw_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(*self.display_origin, image=self.photo, anchor=tk.NW)

        for point in self.detected:
            cx, cy = self._game_to_canvas(point.x, point.y)
            radius = max(7, 15 * self.display_scale)
            color = "#00a6ff" if point.color == "blue" else "#ffc400"
            self.canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                outline=color,
                width=3,
            )

        canvas_points = [self._game_to_canvas(x, y) for x, y, _ in self.selected]
        if len(canvas_points) > 1:
            flattened = [coordinate for point in canvas_points for coordinate in point]
            self.canvas.create_line(*flattened, fill="#ff3366", width=2, dash=(5, 3))
        for index, ((x, y, snapped), (cx, cy)) in enumerate(
            zip(self.selected, canvas_points, strict=True), start=1
        ):
            color = "#00d26a" if snapped else "#ff3366"
            self.canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill=color, outline="white")
            self.canvas.create_text(cx + 10, cy - 10, text=str(index), fill=color, anchor=tk.SW)

    def on_canvas_click(self, event: tk.Event[tk.Misc]) -> None:
        if self.board is None:
            return
        game_point = self._canvas_to_game(event.x, event.y)
        if game_point is None:
            return

        try:
            snap_radius = max(0.0, float(self.snap_var.get()))
        except ValueError:
            messagebox.showerror("参数错误", "吸附半径必须是数字", parent=self.root)
            return

        nearest: Detection | None = None
        nearest_distance = float("inf")
        for point in self.detected:
            px, py = self._game_to_canvas(point.x, point.y)
            distance = float(np.hypot(event.x - px, event.y - py))
            if distance < nearest_distance:
                nearest, nearest_distance = point, distance

        if nearest is not None and nearest_distance <= snap_radius:
            selected = (nearest.x, nearest.y, True)
        else:
            selected = (round(game_point[0], 3), round(game_point[1], 3), False)
        self.selected.append(selected)
        self._refresh_point_list()
        self.render()

    def _game_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        assert self.board is not None
        left, top, right, bottom = self.board
        pixel_x = (x + 25) / 50 * (right - left)
        pixel_y = (15 - y) / 30 * (bottom - top)
        return (
            self.display_origin[0] + pixel_x * self.display_scale,
            self.display_origin[1] + pixel_y * self.display_scale,
        )

    def _canvas_to_game(self, x: float, y: float) -> tuple[float, float] | None:
        assert self.board is not None
        left, top, right, bottom = self.board
        pixel_x = (x - self.display_origin[0]) / self.display_scale
        pixel_y = (y - self.display_origin[1]) / self.display_scale
        width, height = right - left, bottom - top
        if not (0 <= pixel_x <= width and 0 <= pixel_y <= height):
            return None
        return -25 + 50 * pixel_x / width, 15 - 30 * pixel_y / height

    def _refresh_point_list(self) -> None:
        self.point_list.delete(0, tk.END)
        for index, (x, y, snapped) in enumerate(self.selected, start=1):
            suffix = "  [吸附]" if snapped else ""
            self.point_list.insert(tk.END, f"{index}. ({x:.3f}, {y:.3f}){suffix}")

    def undo_point(self) -> None:
        if self.selected:
            self.selected.pop()
            self._refresh_point_list()
            self.render()

    def clear_points(self) -> None:
        self.selected.clear()
        self._refresh_point_list()
        self.render()

    def _generate(self) -> str | None:
        try:
            limit = self.limit_var.get().strip() or None
            return expression_text(
                [(x, y) for x, y, _ in self.selected],
                factor=self.factor_var.get().strip(),
                limit=limit,
                freq=self.freq_var.get().strip(),
                width=self.width_var.get().strip(),
            )
        except (ValueError, TypeError) as exc:
            messagebox.showerror("无法生成表达式", str(exc), parent=self.root)
            return None

    def generate_preview(self) -> None:
        result = self._generate()
        if result is None:
            return
        self.expression.delete("1.0", tk.END)
        self.expression.insert("1.0", result)
        self.status_var.set("表达式已生成")

    def copy_expression(self) -> None:
        result = self._generate()
        if result is None:
            return
        self.expression.delete("1.0", tk.END)
        self.expression.insert("1.0", result)
        self.root.clipboard_clear()
        self.root.clipboard_append(result)
        self.root.update_idletasks()
        self.status_var.set("表达式已复制到剪贴板")


def main() -> None:
    root = tk.Tk()
    GraphwarUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
