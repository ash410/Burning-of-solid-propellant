from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


FORM_END = "Торцевое горение"
FORM_CYLINDER = "Цилиндрический канал"
FORM_STAR = "Звездообразный канал"
FORM_SLOT = "Щелевой заряд"

INSTALL_BONDED = "Скреплённый"
INSTALL_FREE = "Вкладной"

FORMS = (FORM_END, FORM_CYLINDER, FORM_STAR, FORM_SLOT)
INSTALLATIONS = (INSTALL_BONDED, INSTALL_FREE)


@dataclass(slots=True)
class GrainParameters:
    """Геометрия в мм, плотность в кг/м³, скорость горения в мм/с."""

    form: str = FORM_STAR
    installation: str = INSTALL_BONDED
    diameter_mm: float = 240.0
    length_mm: float = 1466.916
    web_mm: float = 50.0
    channel_diameter_override_mm: float | None = None
    fuel_name: str = "Учебное смесевое (методичка)"
    density_kg_m3: float = 1750.0
    burn_rate_mm_s: float = 8.0
    star_points: int = 8
    star_half_angle_deg: float = 32.0
    star_root_radius_mm: float = 2.5
    slot_count: int = 4
    slot_depth_mm: float = 450.0
    slot_width_mm: float = 10.0

    def validate(self) -> None:
        if self.form not in FORMS:
            raise ValueError(f"Неизвестная форма заряда: {self.form}")
        if self.installation not in INSTALLATIONS:
            raise ValueError(f"Неизвестный способ установки: {self.installation}")
        for label, value in (
            ("наружный диаметр", self.diameter_mm),
            ("длина", self.length_mm),
            ("плотность", self.density_kg_m3),
            ("скорость горения", self.burn_rate_mm_s),
        ):
            if value <= 0:
                raise ValueError(f"Параметр «{label}» должен быть положительным")
        if self.form != FORM_END:
            if self.physical_web_mm <= 0 or self.physical_web_mm >= self.diameter_mm / 2:
                raise ValueError("Толщина свода должна быть больше 0 и меньше радиуса заряда")
        if self.form == FORM_CYLINDER and self.channel_diameter_override_mm is not None:
            if not 0 < self.channel_diameter_override_mm < self.diameter_mm:
                raise ValueError("Диаметр канала должен быть больше 0 и меньше наружного диаметра")
        if self.form == FORM_STAR:
            if self.star_points < 3:
                raise ValueError("У звезды должно быть не менее трёх лучей")
            alpha_deg = 180.0 / self.star_points
            if not alpha_deg <= self.star_half_angle_deg < 90.0:
                raise ValueError(
                    f"Полуугол луча β должен лежать в диапазоне [{alpha_deg:.2f}°, 90°)"
                )
            if self.star_root_radius_mm <= 0:
                raise ValueError("Радиус скругления звезды должен быть положительным")
            if self._star_R_mm() <= 0:
                raise ValueError("Для заданного свода и скругления не остаётся геометрии звезды")
            if self.star_stage1_end_mm() < 0:
                raise ValueError("Комбинация β и радиуса скругления даёт отрицательную первую стадию")
        if self.form == FORM_SLOT:
            if self.slot_count < 2:
                raise ValueError("Количество щелей должно быть не меньше двух")
            if not 0 < self.slot_depth_mm < self.length_mm:
                raise ValueError("Глубина щелей должна быть меньше длины заряда")
            channel_d = self.channel_diameter_mm
            if not 0 < self.slot_width_mm < channel_d:
                raise ValueError("Ширина щели должна быть меньше диаметра канала")

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0

    @property
    def channel_diameter_mm(self) -> float:
        if self.form == FORM_CYLINDER and self.channel_diameter_override_mm is not None:
            return self.channel_diameter_override_mm
        return self.diameter_mm - 2.0 * self.web_mm

    @property
    def channel_radius_mm(self) -> float:
        return self.channel_diameter_mm / 2.0

    @property
    def physical_web_mm(self) -> float:
        if self.form == FORM_CYLINDER and self.channel_diameter_override_mm is not None:
            return (self.diameter_mm - self.channel_diameter_override_mm) / 2.0
        return self.web_mm

    def _star_R_mm(self) -> float:
        return self.radius_mm - self.web_mm - self.star_root_radius_mm

    def star_stage1_end_mm(self) -> float:
        alpha = math.pi / self.star_points
        beta = math.radians(self.star_half_angle_deg)
        return self._star_R_mm() * math.sin(alpha) / math.cos(beta) - self.star_root_radius_mm

    def star_complete_burn_mm(self) -> float:
        alpha = math.pi / self.star_points
        R = self._star_R_mm()
        outer = self.radius_mm
        r = self.star_root_radius_mm
        return math.sqrt(R * R + outer * outer - 2.0 * R * outer * math.cos(alpha)) - r

    def burn_limit_mm(self) -> float:
        if self.form == FORM_END:
            return self.length_mm
        if self.installation == INSTALL_FREE:
            # Для сложных каналов расчёт вкладного заряда заканчивается при
            # первом смыкании внутреннего и внешнего фронтов.
            return self.physical_web_mm / 2.0
        if self.form == FORM_STAR:
            return self.star_complete_burn_mm()
        return self.physical_web_mm

    def warnings(self) -> list[str]:
        notes: list[str] = []
        if self.form == FORM_END:
            notes.append("Для торцевого заряда боковая поверхность считается забронированной.")
        if self.installation == INSTALL_FREE and self.form in (FORM_STAR, FORM_SLOT):
            notes.append(
                "Вкладной сложнопрофильный заряд рассчитан до первого смыкания фронтов; "
                "последующий распад и догорание отдельных остатков не моделируются."
            )
        if self.installation == INSTALL_FREE and self.form == FORM_SLOT:
            notes.append(
                "Площадь наружной поверхности щелевого вкладного заряда получена "
                "суперпозицией цилиндрического наружного фронта."
            )
        return notes


@dataclass(slots=True)
class BurnCurve:
    web_mm: np.ndarray
    area_m2: np.ndarray
    components_m2: Dict[str, np.ndarray]
    time_s: np.ndarray
    average_area_m2: float
    initial_mass_kg: float
    burn_limit_mm: float
    burn_duration_s: float
    warnings: list[str]

    def summary(self) -> dict:
        return {
            "burn_limit_mm": self.burn_limit_mm,
            "burn_duration_s": self.burn_duration_s,
            "initial_area_m2": float(self.area_m2[0]),
            "average_area_m2": self.average_area_m2,
            "initial_mass_kg": self.initial_mass_kg,
            "warnings": self.warnings,
        }


def load_fuels(path: Path | None = None) -> dict:
    path = path or Path(__file__).with_name("fuels.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _star_perimeter_mm(e_mm: float, p: GrainParameters) -> float:
    n = p.star_points
    alpha = math.pi / n
    beta = math.radians(p.star_half_angle_deg)
    gamma1 = math.pi / 2.0 + alpha - beta
    r = p.star_root_radius_mm
    R = p._star_R_mm()
    outer = p.radius_mm
    e1 = p.star_stage1_end_mm()
    if e_mm <= e1 + 1e-10:
        b = R * math.sin(alpha) / math.sin(beta) - (r + e_mm) / math.tan(beta)
        return max(0.0, 2.0 * n * (gamma1 * (r + e_mm) + b))
    gamma = alpha + math.asin(_clamp(R * math.sin(alpha) / (r + e_mm)))
    if e_mm <= p.web_mm + 1e-10:
        return max(0.0, 2.0 * n * gamma * (r + e_mm))
    acos_arg = (R * R + (r + e_mm) ** 2 - outer**2) / (2.0 * R * (r + e_mm))
    delta = math.pi - math.acos(_clamp(acos_arg))
    return max(0.0, 2.0 * n * (gamma - delta) * (r + e_mm))


def _slot_components_m2(e_mm: float, p: GrainParameters) -> dict[str, float]:
    # Формулы стр. 9-10 методички. Расчёт ведётся в метрах.
    e = e_mm / 1000.0
    D = p.diameter_mm / 1000.0
    d = p.channel_diameter_mm / 1000.0
    length = p.length_mm / 1000.0
    a = p.slot_depth_mm / 1000.0
    c = p.slot_width_mm / 1000.0
    n = p.slot_count
    alpha = math.pi / n
    e1 = 0.5 * (d * math.sin(alpha) - c) / (1.0 - math.sin(alpha))
    e2 = 0.5 * (D * math.sin(alpha) - c)
    channel = math.pi * (d + 2.0 * e) * max(length - a - e, 0.0)
    beta_arg = _clamp((c + 2.0 * e) / (d + 2.0 * e))
    beta = math.asin(beta_arg)
    channel_slots = (
        n * max(alpha - beta, 0.0) * (d + 2.0 * e) * a if e <= e1 else 0.0
    )
    if e <= e1:
        height = 0.5 * (
            math.sqrt(max(D * D - (c + 2.0 * e) ** 2, 0.0))
            - math.sqrt(max((d + 2.0 * e) ** 2 - (c + 2.0 * e) ** 2, 0.0))
        )
    elif e <= e2:
        height = 0.5 * (
            math.sqrt(max(D * D - (c + 2.0 * e) ** 2, 0.0))
            - (c + 2.0 * e) / math.tan(alpha)
        )
    else:
        height = 0.0
    slot_sides = max(0.0, 2.0 * n * a * height)
    end_face = math.pi / 4.0 * max(D * D - (d + 2.0 * e) ** 2, 0.0)
    return {
        "Канал (I)": channel,
        "Канал у щелей (II)": channel_slots,
        "Боковые поверхности щелей (III)": slot_sides,
        "Торец (IV)": end_face,
    }


def burning_area_components(e_mm: float, p: GrainParameters) -> dict[str, float]:
    p.validate()
    e_mm = min(max(e_mm, 0.0), p.burn_limit_mm())
    D = p.diameter_mm / 1000.0
    length = p.length_mm / 1000.0
    e = e_mm / 1000.0
    if p.form == FORM_END:
        return {"Горящий торец": math.pi * D * D / 4.0}
    if p.form == FORM_CYLINDER:
        inner_d = p.channel_diameter_mm / 1000.0 + 2.0 * e
        result = {"Внутренний канал": math.pi * inner_d * length}
        if p.installation == INSTALL_FREE:
            outer_d = max(D - 2.0 * e, 0.0)
            result["Наружная поверхность"] = math.pi * outer_d * length
        return result
    if p.form == FORM_STAR:
        result = {"Звездообразный канал": _star_perimeter_mm(e_mm, p) / 1000.0 * length}
        if p.installation == INSTALL_FREE:
            result["Наружная поверхность"] = math.pi * max(D - 2.0 * e, 0.0) * length
        return result
    result = _slot_components_m2(e_mm, p)
    if p.installation == INSTALL_FREE:
        result["Наружная поверхность (оценка)"] = (
            math.pi * max(D - 2.0 * e, 0.0) * max(length - e, 0.0)
        )
    return result


def _star_channel_area_m2(p: GrainParameters) -> float:
    n = p.star_points
    alpha = math.pi / n
    beta = math.radians(p.star_half_angle_deg)
    gamma1 = math.pi / 2.0 + alpha - beta
    r = p.star_root_radius_mm / 1000.0
    R = p._star_R_mm() / 1000.0
    b = R * math.sin(alpha) / math.sin(beta) - r / math.tan(beta)
    return n * (gamma1 * r * r + b * (r + R * math.cos(gamma1)) + R * r * math.sin(gamma1))


def _slot_cut_area_m2(p: GrainParameters) -> float:
    D = p.diameter_mm / 1000.0
    d = p.channel_diameter_mm / 1000.0
    c = p.slot_width_mm / 1000.0
    term_outer = (
        c / 2.0 * math.sqrt(max(D * D / 4.0 - c * c / 4.0, 0.0))
        + D * D / 4.0 * math.asin(_clamp(c / D))
    )
    term_inner = (
        c / 2.0 * math.sqrt(max(d * d / 4.0 - c * c / 4.0, 0.0))
        + d * d / 4.0 * math.asin(_clamp(c / d))
    )
    return p.slot_count * (term_outer - term_inner)


def initial_fuel_volume_m3(p: GrainParameters) -> float:
    p.validate()
    D = p.diameter_mm / 1000.0
    length = p.length_mm / 1000.0
    chamber_area = math.pi * D * D / 4.0
    if p.form == FORM_END:
        return chamber_area * length
    if p.form == FORM_CYLINDER:
        d = p.channel_diameter_mm / 1000.0
        return math.pi / 4.0 * (D * D - d * d) * length
    if p.form == FORM_STAR:
        return (chamber_area - _star_channel_area_m2(p)) * length
    d = p.channel_diameter_mm / 1000.0
    annulus_volume = math.pi / 4.0 * (D * D - d * d) * length
    return annulus_volume - _slot_cut_area_m2(p) * p.slot_depth_mm / 1000.0


def calculate_curve(p: GrainParameters, samples: int = 1001) -> BurnCurve:
    p.validate()
    if samples < 3:
        raise ValueError("Для графика требуется не менее трёх точек")
    limit = p.burn_limit_mm()
    web = np.linspace(0.0, limit, samples)
    rows = [burning_area_components(float(e), p) for e in web]
    component_names: Iterable[str] = dict.fromkeys(k for row in rows for k in row)
    components = {
        name: np.array([row.get(name, 0.0) for row in rows], dtype=float)
        for name in component_names
    }
    total = np.sum(np.vstack(tuple(components.values())), axis=0)
    average = float(np.trapezoid(total, web) / limit)
    duration = limit / p.burn_rate_mm_s
    return BurnCurve(
        web_mm=web,
        area_m2=total,
        components_m2=components,
        time_s=web / p.burn_rate_mm_s,
        average_area_m2=average,
        initial_mass_kg=initial_fuel_volume_m3(p) * p.density_kg_m3,
        burn_limit_mm=limit,
        burn_duration_s=duration,
        warnings=p.warnings(),
    )


def parameters_as_dict(p: GrainParameters) -> dict:
    return asdict(p)
