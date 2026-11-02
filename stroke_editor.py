#!/usr/bin/env python3
"""Create character stroke trajectories by clicking points on a canvas."""

import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageFont


CANVAS_SIZE = 512
DATA_SIZE = 256
TARGET_FONT_SIZE = 184
FONT_PATH = Path(r"C:\Windows\Fonts\HGRSKP.TTF")
BRUSH_WIDTH = 6
PRESSURE = 0.65
STROKE_COLORS = ("#1769aa", "#d84315", "#388e3c", "#8e24aa", "#ef6c00")


class StrokeEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("文字ストローク作成")
        self.root.resizable(False, False)
        self.strokes = [[]]
        self.character = tk.StringVar(value="一")
        self.status = tk.StringVar(value="キャンバスをクリックして点を追加してください。")
        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            background="white",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        self._build_ui()
        self._draw_canvas()

    def _build_ui(self):
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(fill="x")

        ttk.Label(controls, text="表示する文字").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(controls, textvariable=self.character, width=8)
        entry.grid(row=0, column=1, padx=(8, 6))
        entry.bind("<Return>", lambda _event: self._update_character())
        ttk.Button(
            controls, text="表示", command=self._update_character
        ).grid(row=0, column=2)
        ttk.Button(
            controls, text="新しいストローク", command=self._new_stroke
        ).grid(row=0, column=3, padx=(12, 0))
        ttk.Button(controls, text="1点戻す", command=self._undo).grid(
            row=0, column=4, padx=(6, 0)
        )
        ttk.Button(controls, text="すべて消去", command=self._clear).grid(
            row=0, column=5, padx=(6, 0)
        )
        ttk.Button(controls, text="CSVに保存", command=self._save).grid(
            row=0, column=6, padx=(12, 0)
        )

        self.canvas.pack(padx=12, pady=(0, 8))
        self.canvas.bind("<Button-1>", self._add_point)
        ttk.Label(
            self.root,
            textvariable=self.status,
            padding=(12, 0, 12, 12),
            wraplength=CANVAS_SIZE,
        ).pack(fill="x")

    def _update_character(self):
        value = self.character.get()
        if len(value) != 1:
            messagebox.showerror("入力エラー", "表示する文字は1文字にしてください。")
            return
        self._draw_canvas()

    def _draw_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            CANVAS_SIZE / 2,
            CANVAS_SIZE / 2,
            text=self.character.get(),
            fill="#e4e8ed",
            font=("HG正楷書体-PRO", 368),
            anchor="center",
        )
        self.canvas.create_rectangle(
            0, 0, CANVAS_SIZE - 1, CANVAS_SIZE - 1, outline="#cbd5e1"
        )

        for stroke_index, points in enumerate(self.strokes):
            color = STROKE_COLORS[stroke_index % len(STROKE_COLORS)]
            if len(points) > 1:
                self.canvas.create_line(
                    *[coordinate for point in points for coordinate in point],
                    fill=color,
                    width=BRUSH_WIDTH,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                )
            for point_index, (x, y) in enumerate(points):
                radius = 7
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=color,
                    outline="white",
                    width=2,
                )
                self.canvas.create_text(
                    x + 11,
                    y - 11,
                    text=str(point_index + 1),
                    fill=color,
                    font=("Yu Gothic UI", 10, "bold"),
                    anchor="sw",
                )

        count = sum(len(points) for points in self.strokes)
        stroke_count = sum(bool(points) for points in self.strokes)
        self.status.set(
            f"ストローク: {stroke_count}  点: {count}  "
            "クリックで点を追加し、「新しいストローク」で区切ります。"
        )

    def _add_point(self, event):
        x = max(0, min(CANVAS_SIZE - 1, event.x))
        y = max(0, min(CANVAS_SIZE - 1, event.y))
        self.strokes[-1].append((x, y))
        self._draw_canvas()

    def _new_stroke(self):
        if self.strokes[-1]:
            self.strokes.append([])
            self._draw_canvas()

    def _undo(self):
        while len(self.strokes) > 1 and not self.strokes[-1]:
            self.strokes.pop()
        if self.strokes[-1]:
            self.strokes[-1].pop()
        self._draw_canvas()

    def _clear(self):
        self.strokes = [[]]
        self._draw_canvas()

    def _save(self):
        character = self.character.get()
        if len(character) != 1:
            messagebox.showerror("入力エラー", "表示する文字は1文字にしてください。")
            return
        populated_strokes = [points for points in self.strokes if points]
        if not populated_strokes:
            messagebox.showerror("保存エラー", "保存する点がありません。")
            return

        output_path = filedialog.asksaveasfilename(
            title="ストローク情報を保存",
            initialdir=Path("artifacts") / "handdrawn",
            initialfile=f"{character}.csv",
            defaultextension=".csv",
            filetypes=(("CSVファイル", "*.csv"), ("すべてのファイル", "*.*")),
        )
        if not output_path:
            return

        if not FONT_PATH.is_file():
            messagebox.showerror(
                "保存エラー",
                f"HG正楷書体-PRO のフォントファイルが見つかりません。\n{FONT_PATH}",
            )
            return
        try:
            target = Image.new("L", (DATA_SIZE, DATA_SIZE), 255)
            draw = ImageDraw.Draw(target)
            font = ImageFont.truetype(str(FONT_PATH), TARGET_FONT_SIZE)
            draw.text(
                (DATA_SIZE / 2, DATA_SIZE / 2),
                character,
                font=font,
                fill=0,
                anchor="mm",
            )
            target = target.point(lambda value: 0 if value < 192 else 255)
        except (OSError, ValueError) as error:
            messagebox.showerror("画像生成エラー", f"target.pngを生成できませんでした。\n{error}")
            return

        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("stroke_id", "progress", "x_px", "y_px", "z_press"))
                for stroke_id, points in enumerate(populated_strokes, start=1):
                    for index, (canvas_x, canvas_y) in enumerate(points):
                        progress = index / (len(points) - 1) if len(points) > 1 else 0.0
                        x_px = round(canvas_x * (DATA_SIZE - 1) / (CANVAS_SIZE - 1))
                        y_px = round(canvas_y * (DATA_SIZE - 1) / (CANVAS_SIZE - 1))
                        writer.writerow(
                            (stroke_id, progress, x_px, y_px, PRESSURE)
                        )
            target.save(path.with_suffix(".png"))
        except OSError as error:
            messagebox.showerror(
                "保存エラー",
                f"CSVまたはtarget.pngを保存できませんでした。\n{error}",
            )
            return

        self.status.set(f"保存しました: {path}")


def main():
    root = tk.Tk()
    StrokeEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
