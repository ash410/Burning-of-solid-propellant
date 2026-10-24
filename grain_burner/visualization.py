from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle, Wedge
import numpy as np

from .model import (
    BurnCurve,
    FORM_CYLINDER,
    FORM_END,
    FORM_SLOT,
    FORM_STAR,
    INSTALL_FREE,
    GrainParameters,
    calculate_curve,
)


FUEL_COLOR = "#e8872f"
FUEL_LIGHT = "#ffd19a"
CHAMBER_COLOR = "#29313d"
FLAME_COLOR = "#d13c2f"
GRID_COLOR = "#d8dde5"


@dataclass(slots=True)
class GeometryCache:
    x_mm: np.ndarray
    y_mm: np.ndarray
    inside_initial_void: np.ndarray
    distance_to_void_mm: np.ndarray
    extent: tuple[float, float, float, float]


def _point_in_polygon(
    x: np.ndarray, y: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    inside = np.zeros_like(x, dtype=bool)
    xj, yj = vertices[-1]
    for xi, yi in vertices:
        crossing = (yi > y) != (yj > y)
        x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
        inside ^= crossing & (x < x_cross)
        xj, yj = xi, yi
    return inside


def _distance_to_segments(
    x: np.ndarray, y: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    distance = np.full_like(x, np.inf, dtype=float)
    previous = vertices[-1]
    for current in vertices:
        ax, ay = previous
        bx, by = current
        dx, dy = bx - ax, by - ay
        scale = dx * dx + dy * dy
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / scale, 0.0, 1.0)
        projection_x = ax + t * dx
        projection_y = ay + t * dy
        distance = np.minimum(distance, np.hypot(x - projection_x, y - projection_y))
        previous = current
    return distance


def _star_vertices(p: GrainParameters) -> np.ndarray:
    count = p.star_points
    alpha = np.pi / count
    beta = np.radians(p.star_half_angle_deg)
    R = p._star_R_mm()
    tip_radius = p.radius_mm - p.web_mm
    root_radius = max(
        0.08 * p.radius_mm,
        R * np.sin(alpha) / np.sin(beta) - p.star_root_radius_mm,
    )
    angles = np.pi / 2.0 + np.arange(2 * count) * np.pi / count
    radii = np.where(np.arange(2 * count) % 2 == 0, tip_radius, root_radius)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def _rectangle_distance(
    u: np.ndarray,
    v: np.ndarray,
    u_min: float,
    u_max: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    du = np.maximum(np.maximum(u_min - u, 0.0), u - u_max)
    dv = np.maximum(np.abs(v) - half_width, 0.0)
    inside = (u >= u_min) & (u <= u_max) & (np.abs(v) <= half_width)
    return np.hypot(du, dv), inside


def build_geometry_cache(p: GrainParameters, resolution: int = 361) -> GeometryCache | None:
    if p.form in (FORM_END, FORM_CYLINDER):
        return None
    radius = p.radius_mm
    axis = np.linspace(-radius, radius, resolution)
    x, y = np.meshgrid(axis, axis)
    if p.form == FORM_STAR:
        vertices = _star_vertices(p)
        inside_void = _point_in_polygon(x, y, vertices)
        distance = _distance_to_segments(x, y, vertices)
    else:
        rho = np.hypot(x, y)
        inner = p.channel_radius_mm
        inside_void = rho <= inner
        distance = np.maximum(rho - inner, 0.0)
        for index in range(p.slot_count):
            angle = 2.0 * np.pi * index / p.slot_count
            u = x * np.cos(angle) + y * np.sin(angle)
            v = -x * np.sin(angle) + y * np.cos(angle)
            rect_distance, inside_rect = _rectangle_distance(
                u, v, inner, radius, p.slot_width_mm / 2.0
            )
            distance = np.minimum(distance, rect_distance)
            inside_void |= inside_rect
    return GeometryCache(
        x_mm=x,
        y_mm=y,
        inside_initial_void=inside_void,
        distance_to_void_mm=distance,
        extent=(-radius, radius, -radius, radius),
    )


def remaining_mask(
    p: GrainParameters, e_mm: float, cache: GeometryCache
) -> np.ndarray:
    if p.installation != INSTALL_FREE and e_mm >= p.burn_limit_mm() - 1e-9:
        return np.zeros_like(cache.inside_initial_void, dtype=bool)
    rho = np.hypot(cache.x_mm, cache.y_mm)
    outer_radius = p.radius_mm - (e_mm if p.installation == INSTALL_FREE else 0.0)
    return (
        (rho <= outer_radius)
        & (~cache.inside_initial_void)
        & (cache.distance_to_void_mm >= e_mm)
    )


def _draw_slot_views(
    ax,
    p: GrainParameters,
    e_mm: float,
    cache: GeometryCache,
) -> None:
    """Показывает одновременно зону щелей и регрессию горящего торца."""
    radius = p.radius_mm
    offset = 1.18 * radius
    mask = remaining_mask(p, e_mm, cache)
    cmap = ListedColormap(["#f7f8fa", FUEL_LIGHT])
    shifted_extent = (
        cache.extent[0] - offset,
        cache.extent[1] - offset,
        cache.extent[2],
        cache.extent[3],
    )
    ax.imshow(
        mask.astype(int),
        origin="lower",
        extent=shifted_extent,
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    if np.any(mask) and np.any(~mask):
        ax.contour(
            cache.x_mm - offset,
            cache.y_mm,
            mask.astype(float),
            levels=[0.5],
            colors=[FUEL_COLOR],
            linewidths=1.05,
        )
    ax.add_patch(
        Circle((-offset, 0.0), radius, fill=False, color=CHAMBER_COLOR, linewidth=1.4)
    )

    outer = radius - (e_mm if p.installation == INSTALL_FREE else 0.0)
    inner = p.channel_radius_mm + e_mm
    alpha = np.pi / p.slot_count
    slot_burnout = 0.5 * (p.diameter_mm * np.sin(alpha) - p.slot_width_mm)
    end_edge = FLAME_COLOR if e_mm >= slot_burnout else FUEL_COLOR
    if outer > inner:
        ax.add_patch(
            Wedge(
                (offset, 0.0),
                outer,
                0.0,
                360.0,
                width=outer - inner,
                facecolor=FUEL_LIGHT,
                edgecolor=end_edge,
                linewidth=2.0 if e_mm >= slot_burnout else 1.4,
            )
        )
    ax.add_patch(
        Circle((offset, 0.0), radius, fill=False, color=CHAMBER_COLOR, linewidth=1.4)
    )
    ax.text(-offset, radius * 1.12, "Зона щелей", ha="center", va="bottom", fontsize=9)
    ax.text(offset, radius * 1.12, "Торцевой фронт", ha="center", va="bottom", fontsize=9)
    if e_mm >= slot_burnout:
        status = "Щели сгорели — продолжается горение канала и торца"
    else:
        status = "Совместное горение щелей, канала и торца"
    ax.text(0.0, -radius * 1.18, status, ha="center", va="top", fontsize=8, color="#384657")
    ax.set_xlim(-2.25 * radius, 2.25 * radius)
    ax.set_ylim(-1.32 * radius, 1.32 * radius)


def _style_geometry_axis(ax, p: GrainParameters, e_mm: float) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f7f8fa")
    ax.grid(False)
    ax.set_title(f"Геометрия, e = {e_mm:.1f} мм", fontsize=11)
    ax.set_xlabel("мм")
    ax.set_ylabel("мм")
    ax.tick_params(labelsize=8)


def draw_grain(
    ax,
    p: GrainParameters,
    e_mm: float,
    cache: GeometryCache | None = None,
) -> None:
    ax.clear()
    _style_geometry_axis(ax, p, e_mm)
    radius = p.radius_mm
    if p.form == FORM_END:
        remaining_length = max(p.length_mm - e_mm, 0.0)
        ax.add_patch(
            Rectangle(
                (e_mm, -radius),
                remaining_length,
                2.0 * radius,
                facecolor=FUEL_LIGHT,
                edgecolor=FUEL_COLOR,
                linewidth=1.7,
            )
        )
        ax.axvline(e_mm, color=FLAME_COLOR, linewidth=2.5, label="фронт горения")
        ax.set_xlim(-0.03 * p.length_mm, 1.03 * p.length_mm)
        ax.set_ylim(-1.25 * radius, 1.25 * radius)
        ax.set_xlabel("длина заряда, мм")
        ax.legend(loc="upper right", fontsize=8)
        return
    if p.form == FORM_CYLINDER:
        inner = p.channel_radius_mm + e_mm
        outer = radius - (e_mm if p.installation == INSTALL_FREE else 0.0)
        if outer > inner:
            ax.add_patch(
                Wedge(
                    (0.0, 0.0),
                    outer,
                    0.0,
                    360.0,
                    width=outer - inner,
                    facecolor=FUEL_LIGHT,
                    edgecolor=FUEL_COLOR,
                    linewidth=1.7,
                )
            )
        ax.add_patch(Circle((0, 0), radius, fill=False, color=CHAMBER_COLOR, linewidth=1.5))
    elif p.form == FORM_SLOT:
        if cache is None:
            cache = build_geometry_cache(p)
        assert cache is not None
        _draw_slot_views(ax, p, e_mm, cache)
        return
    else:
        if cache is None:
            cache = build_geometry_cache(p)
        assert cache is not None
        mask = remaining_mask(p, e_mm, cache)
        cmap = ListedColormap(["#f7f8fa", FUEL_LIGHT])
        ax.imshow(
            mask.astype(int),
            origin="lower",
            extent=cache.extent,
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        if np.any(mask) and np.any(~mask):
            ax.contour(
                cache.x_mm,
                cache.y_mm,
                mask.astype(float),
                levels=[0.5],
                colors=[FUEL_COLOR],
                linewidths=1.15,
            )
        ax.add_patch(Circle((0, 0), radius, fill=False, color=CHAMBER_COLOR, linewidth=1.5))
    padding = radius * 1.08
    ax.set_xlim(-padding, padding)
    ax.set_ylim(-padding, padding)


def draw_curve(ax, p: GrainParameters, curve: BurnCurve, e_mm: float | None = None):
    ax.clear()
    ax.plot(curve.web_mm, curve.area_m2, color=FLAME_COLOR, linewidth=2.2, label="Суммарная площадь")
    if p.form == FORM_SLOT:
        for name, values in curve.components_m2.items():
            ax.plot(curve.web_mm, values, linewidth=1.0, alpha=0.72, label=name)
    ax.axhline(
        curve.average_area_m2,
        color="#3b82b7",
        linestyle="--",
        linewidth=1.2,
        label=f"Средняя: {curve.average_area_m2:.3f} м²",
    )
    marker = None
    if e_mm is not None:
        area = float(np.interp(e_mm, curve.web_mm, curve.area_m2))
        marker = ax.scatter([e_mm], [area], s=55, color="#202733", zorder=5)
        ax.axvline(e_mm, color="#202733", alpha=0.25, linewidth=1.0)
    ax.set_title("Площадь поверхности горения", fontsize=11)
    ax.set_xlabel("Толщина сгоревшего свода e, мм")
    ax.set_ylabel("Sг, м²")
    ax.set_xlim(0.0, curve.burn_limit_mm)
    y_max = max(float(np.max(curve.area_m2)), 1e-6)
    ax.set_ylim(0.0, 1.12 * y_max)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.legend(loc="best", fontsize=7)
    ax.tick_params(labelsize=8)
    return marker


def make_figure(
    p: GrainParameters,
    curve: BurnCurve | None = None,
    e_mm: float = 0.0,
    figsize: tuple[float, float] = (11.2, 5.2),
):
    curve = curve or calculate_curve(p)
    cache = build_geometry_cache(p)
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    draw_grain(axes[0], p, e_mm, cache)
    draw_curve(axes[1], p, curve, e_mm)
    fig.suptitle(
        f"{p.form} — {p.installation.lower()} заряд | "
        f"t = {e_mm / p.burn_rate_mm_s:.2f} с",
        fontsize=13,
        fontweight="semibold",
    )
    return fig, axes, cache, curve


def save_static_plot(p: GrainParameters, path: str | Path, dpi: int = 170) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    curve = calculate_curve(p)
    fig, _, _, _ = make_figure(p, curve=curve, e_mm=0.0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_animation(
    p: GrainParameters,
    path: str | Path,
    frame_count: int = 70,
    fps: int = 60,
    dpi: int = 105,
    playback_speed: float = 1.0,
) -> Path:
    if playback_speed <= 0:
        raise ValueError("Скорость воспроизведения должна быть положительной")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    curve = calculate_curve(p)
    cache = build_geometry_cache(p)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), constrained_layout=True)
    output_frame_count = max(2, int(round(frame_count / playback_speed)))
    frame_web = np.linspace(0.0, curve.burn_limit_mm, output_frame_count)

    def update(frame_index: int):
        e_mm = float(frame_web[frame_index])
        draw_grain(axes[0], p, e_mm, cache)
        marker = draw_curve(axes[1], p, curve, e_mm)
        fig.suptitle(
            f"{p.form} — {p.installation.lower()} заряд | "
            f"e = {e_mm:.1f} мм, t = {e_mm / p.burn_rate_mm_s:.2f} с",
            fontsize=12,
            fontweight="semibold",
        )
        return (marker,) if marker is not None else ()

    animation = FuncAnimation(
        fig, update, frames=output_frame_count, interval=1000 / fps, blit=False
    )
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
