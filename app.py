from __future__ import annotations

import json
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

PROJECT_DIR = Path(__file__).resolve().parent
ANIMATION_FPS = 60
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from grain_burner.model import (
    FORMS,
    INSTALLATIONS,
    FORM_CYLINDER,
    FORM_END,
    FORM_SLOT,
    FORM_STAR,
    INSTALL_BONDED,
    GrainParameters,
    calculate_curve,
    load_fuels,
)
from grain_burner.visualization import (
    build_geometry_cache,
    draw_curve,
    draw_grain,
    save_animation,
)


class GrainBurnerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Геометрическое горение зарядов РДТТ")
        self.geometry("1450x850")
        self.minsize(1160, 720)
        self.option_add("*Font", ("Arial", 10))
        self.fuels = load_fuels()
        self.entries: dict[str, ttk.Entry] = {}
        self.curve = None
        self.params = None
        self.geometry_cache = None
        self.animation_web = np.array([0.0])
        self.animation_index = 0
        self.animation_position = 0.0
        self.animation_running = False
        self.after_id: str | None = None
        self.next_frame_deadline = 0.0
        self.animation_speed_var = tk.DoubleVar(value=1.0)
        self.animation_speed_text = tk.StringVar(value="1.00×")
        self._build_ui()
        self.apply_manual_example(recalculate=False)
        self.calculate()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=(12, 10))
        controls.grid(row=0, column=0, sticky="nsw")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Параметры заряда", font=("Arial", 15, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )
        row = 1
        self.form_var = tk.StringVar(value=FORM_STAR)
        self.installation_var = tk.StringVar(value=INSTALL_BONDED)
        self.fuel_var = tk.StringVar(value=next(iter(self.fuels)))
        self.form_box = self._combo(controls, row, "Форма", self.form_var, FORMS)
        row += 1
        self.installation_box = self._combo(
            controls, row, "Установка", self.installation_var, INSTALLATIONS
        )
        row += 1
        self.fuel_box = self._combo(
            controls, row, "Марка топлива", self.fuel_var, tuple(self.fuels)
        )
        row += 1
        self.form_box.bind("<<ComboboxSelected>>", self._on_form_change)
        self.fuel_box.bind("<<ComboboxSelected>>", self._on_fuel_change)

        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(controls, text="Основная геометрия", font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        row += 1
        for key, label, value in (
            ("diameter_mm", "Диаметр D, мм", "240"),
            ("channel_diameter_mm", "Диаметр канала d, мм", "140"),
            ("length_mm", "Длина L, мм", "1466.916"),
            ("web_mm", "Физический свод, мм", "50"),
            ("density_kg_m3", "Плотность, кг/м³", "1750"),
            ("burn_rate_mm_s", "Скорость горения, мм/с", "8"),
            ("frame_count", "Кадров анимации", "180"),
        ):
            self._entry(controls, row, key, label, value)
            row += 1

        ttk.Label(controls, text="Скорость анимации").grid(
            row=row, column=0, sticky="w", pady=4
        )
        speed_frame = ttk.Frame(controls)
        speed_frame.grid(row=row, column=1, sticky="ew", pady=4, padx=(8, 0))
        speed_frame.columnconfigure(0, weight=1)
        self.speed_scale = ttk.Scale(
            speed_frame,
            from_=0.25,
            to=4.0,
            variable=self.animation_speed_var,
            command=self._on_speed_change,
        )
        self.speed_scale.grid(row=0, column=0, sticky="ew")
        ttk.Label(speed_frame, textvariable=self.animation_speed_text, width=6).grid(
            row=0, column=1, padx=(6, 0)
        )
        row += 1

        self.star_frame = ttk.LabelFrame(controls, text="Звездообразный канал", padding=(8, 4))
        self.star_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self.star_frame.columnconfigure(1, weight=1)
        star_row = 0
        for key, label, value in (
            ("star_points", "Число лучей n", "8"),
            ("star_half_angle_deg", "Полуугол β, град", "32"),
            ("star_root_radius_mm", "Скругление r, мм", "2.5"),
        ):
            self._entry(self.star_frame, star_row, key, label, value)
            star_row += 1
        row += 1

        self.slot_frame = ttk.LabelFrame(controls, text="Щелевой заряд", padding=(8, 4))
        self.slot_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self.slot_frame.columnconfigure(1, weight=1)
        slot_row = 0
        for key, label, value in (
            ("slot_count", "Количество щелей n", "4"),
            ("slot_depth_mm", "Глубина щелей a, мм", "450"),
            ("slot_width_mm", "Ширина щелей c, мм", "10"),
        ):
            self._entry(self.slot_frame, slot_row, key, label, value)
            slot_row += 1
        row += 1
        self._update_parameter_visibility()

        button_row = ttk.Frame(controls)
        button_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 5))
        ttk.Button(button_row, text="Рассчитать", command=self.calculate).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(button_row, text="Пример методички", command=self.apply_manual_example).pack(
            side="left", fill="x", expand=True, padx=(6, 0)
        )
        row += 1
        animation_row = ttk.Frame(controls)
        animation_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        self.play_button = ttk.Button(animation_row, text="Пауза", command=self.toggle_animation)
        self.play_button.pack(side="left", fill="x", expand=True)
        ttk.Button(animation_row, text="Сначала", command=self.restart_animation).pack(
            side="left", fill="x", expand=True, padx=(6, 0)
        )
        row += 1
        export_row = ttk.Frame(controls)
        export_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(export_row, text="Сохранить PNG", command=self.save_png).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(export_row, text="Сохранить GIF", command=self.save_gif).pack(
            side="left", fill="x", expand=True, padx=(6, 0)
        )
        row += 1

        self.result_var = tk.StringVar()
        result = ttk.Label(
            controls,
            textvariable=self.result_var,
            justify="left",
            wraplength=330,
            foreground="#263241",
        )
        result.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        plot_frame = ttk.Frame(self, padding=(4, 8, 10, 6))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(10.8, 6.4), constrained_layout=True)
        self.ax_geometry, self.ax_curve = self.figure.subplots(1, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame).update()

    def _combo(self, parent, row, label, variable, values):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        box = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=24)
        box.grid(row=row, column=1, sticky="ew", pady=3, padx=(8, 0))
        return box

    def _entry(self, parent, row, key, label, value):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, width=16)
        entry.insert(0, value)
        entry.grid(row=row, column=1, sticky="ew", pady=2, padx=(8, 0))
        self.entries[key] = entry

    def _set_entry(self, key: str, value) -> None:
        self.entries[key].delete(0, tk.END)
        self.entries[key].insert(0, str(value))

    def _on_form_change(self, _event=None) -> None:
        self._update_parameter_visibility()
        self.apply_manual_example()

    def _update_parameter_visibility(self) -> None:
        if self.form_var.get() == FORM_STAR:
            self.star_frame.grid()
        else:
            self.star_frame.grid_remove()
        if self.form_var.get() == FORM_SLOT:
            self.slot_frame.grid()
        else:
            self.slot_frame.grid_remove()

    def _on_fuel_change(self, _event=None) -> None:
        fuel = self.fuels[self.fuel_var.get()]
        self._set_entry("density_kg_m3", fuel["density_kg_m3"])
        self._set_entry("burn_rate_mm_s", fuel["burn_rate_mm_s"])

    def _on_speed_change(self, value: str) -> None:
        speed = float(value)
        self.animation_speed_text.set(f"{speed:.2f}×")

    def apply_manual_example(self, recalculate: bool = True) -> None:
        shape = self.form_var.get()
        values = {
            "diameter_mm": 240,
            "channel_diameter_mm": 140,
            "length_mm": 1466.916,
            "web_mm": 50,
            "star_points": 8,
            "star_half_angle_deg": 32,
            "star_root_radius_mm": 2.5,
            "slot_count": 4,
            "slot_depth_mm": 450,
            "slot_width_mm": 10,
        }
        if shape == FORM_SLOT:
            values.update(length_mm=1385.585, web_mm=90, channel_diameter_mm=60)
        elif shape == FORM_CYLINDER:
            values.update(length_mm=1000, web_mm=50, channel_diameter_mm=140)
        elif shape == FORM_END:
            values.update(length_mm=600, web_mm=50)
        for key, value in values.items():
            self._set_entry(key, value)
        self.installation_var.set(INSTALL_BONDED)
        if recalculate:
            self.calculate()

    def _read_parameters(self) -> GrainParameters:
        def number(key: str) -> float:
            return float(self.entries[key].get().replace(",", "."))

        diameter = number("diameter_mm")
        channel_override = number("channel_diameter_mm") if self.form_var.get() == FORM_CYLINDER else None
        return GrainParameters(
            form=self.form_var.get(),
            installation=self.installation_var.get(),
            diameter_mm=diameter,
            length_mm=number("length_mm"),
            web_mm=number("web_mm"),
            channel_diameter_override_mm=channel_override,
            fuel_name=self.fuel_var.get(),
            density_kg_m3=number("density_kg_m3"),
            burn_rate_mm_s=number("burn_rate_mm_s"),
            star_points=int(number("star_points")),
            star_half_angle_deg=number("star_half_angle_deg"),
            star_root_radius_mm=number("star_root_radius_mm"),
            slot_count=int(number("slot_count")),
            slot_depth_mm=number("slot_depth_mm"),
            slot_width_mm=number("slot_width_mm"),
        )

    def calculate(self) -> None:
        try:
            params = self._read_parameters()
            curve = calculate_curve(params)
            frames = max(10, int(float(self.entries["frame_count"].get())))
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Ошибка параметров", str(exc))
            return
        self.params = params
        self.curve = curve
        self.geometry_cache = build_geometry_cache(params)
        self.animation_web = np.linspace(0.0, curve.burn_limit_mm, frames)
        self.animation_index = 0
        self.animation_position = 0.0
        notes = "\n".join(f"• {note}" for note in curve.warnings)
        self.result_var.set(
            f"Расчётная глубина: {curve.burn_limit_mm:.2f} мм\n"
            f"Физический свод: {params.physical_web_mm:.2f} мм\n"
            f"Время: {curve.burn_duration_s:.2f} с\n"
            f"Sг(0): {curve.area_m2[0]:.4f} м²\n"
            f"Средняя Sг: {curve.average_area_m2:.4f} м²\n"
            f"Начальная масса: {curve.initial_mass_kg:.2f} кг"
            + (f"\n\n{notes}" if notes else "")
        )
        self.animation_running = True
        self.next_frame_deadline = time.perf_counter() + 1.0 / ANIMATION_FPS
        self.play_button.configure(text="Пауза")
        self._render_frame()
        self._schedule_tick()

    def _render_frame(self) -> None:
        if self.params is None or self.curve is None:
            return
        e_mm = float(self.animation_web[self.animation_index])
        draw_grain(self.ax_geometry, self.params, e_mm, self.geometry_cache)
        draw_curve(self.ax_curve, self.params, self.curve, e_mm)
        self.figure.suptitle(
            f"{self.params.form} — {self.params.installation.lower()} заряд | "
            f"e = {e_mm:.1f} мм, t = {e_mm / self.params.burn_rate_mm_s:.2f} с",
            fontsize=13,
            fontweight="semibold",
        )
        self.canvas.draw_idle()

    def _schedule_tick(self) -> None:
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        delay_ms = max(1, round((self.next_frame_deadline - time.perf_counter()) * 1000))
        self.after_id = self.after(delay_ms, self._tick)

    def _tick(self) -> None:
        self.after_id = None
        if not self.animation_running:
            return
        last_index = len(self.animation_web) - 1
        if self.animation_position >= last_index:
            self.animation_running = False
            self.play_button.configure(text="Воспроизвести")
            return
        speed = max(0.01, float(self.animation_speed_var.get()))
        self.animation_position = min(float(last_index), self.animation_position + speed)
        new_index = min(last_index, int(self.animation_position))
        if new_index != self.animation_index:
            self.animation_index = new_index
            self._render_frame()
        self.next_frame_deadline += 1.0 / ANIMATION_FPS
        self._schedule_tick()

    def toggle_animation(self) -> None:
        if self.params is None:
            return
        self.animation_running = not self.animation_running
        self.play_button.configure(text="Пауза" if self.animation_running else "Воспроизвести")
        if self.animation_running:
            if self.animation_index >= len(self.animation_web) - 1:
                self.animation_index = 0
                self.animation_position = 0.0
            self.next_frame_deadline = time.perf_counter() + 1.0 / ANIMATION_FPS
            self._schedule_tick()

    def restart_animation(self) -> None:
        self.animation_index = 0
        self.animation_position = 0.0
        self.animation_running = True
        self.next_frame_deadline = time.perf_counter() + 1.0 / ANIMATION_FPS
        self.play_button.configure(text="Пауза")
        self._render_frame()
        self._schedule_tick()

    def save_png(self) -> None:
        if self.params is None:
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить график",
            defaultextension=".png",
            filetypes=(("PNG", "*.png"),),
            initialfile="burning_area.png",
        )
        if not path:
            return
        self.figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        messagebox.showinfo("Готово", f"График сохранён:\n{path}")

    def save_gif(self) -> None:
        if self.params is None:
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить анимацию",
            defaultextension=".gif",
            filetypes=(("GIF", "*.gif"),),
            initialfile="grain_burning.gif",
        )
        if not path:
            return
        try:
            frames = max(10, int(float(self.entries["frame_count"].get())))
            self.configure(cursor="watch")
            self.update_idletasks()
            save_animation(
                self.params,
                path,
                frame_count=frames,
                fps=ANIMATION_FPS,
                playback_speed=float(self.animation_speed_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("Ошибка экспорта", str(exc))
        else:
            messagebox.showinfo("Готово", f"Анимация сохранена:\n{path}")
        finally:
            self.configure(cursor="")


if __name__ == "__main__":
    GrainBurnerApp().mainloop()
