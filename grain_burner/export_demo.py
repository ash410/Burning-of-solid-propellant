from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplconfig"))

from .model import (  # noqa: E402
    FORM_CYLINDER,
    FORM_END,
    FORM_SLOT,
    FORM_STAR,
    INSTALL_BONDED,
    INSTALL_FREE,
    GrainParameters,
    calculate_curve,
    parameters_as_dict,
)
from .visualization import save_animation, save_static_plot  # noqa: E402


SHAPE_ALIASES = {
    "end": FORM_END,
    "cylinder": FORM_CYLINDER,
    "star": FORM_STAR,
    "slot": FORM_SLOT,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт графика и GIF горения заряда РДТТ")
    parser.add_argument("--shape", choices=SHAPE_ALIASES, default="star")
    parser.add_argument("--installation", choices=("bonded", "free"), default="bonded")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--channel-diameter",
        type=float,
        default=None,
        help="Диаметр цилиндрического канала, мм",
    )
    parser.add_argument("--frames", type=int, default=70)
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Множитель скорости анимации"
    )
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    params = GrainParameters(
        form=SHAPE_ALIASES[args.shape],
        installation=INSTALL_BONDED if args.installation == "bonded" else INSTALL_FREE,
        channel_diameter_override_mm=args.channel_diameter,
    )
    if params.form == FORM_SLOT:
        params.web_mm = 90.0
        params.length_mm = 1385.585
    if params.form == FORM_END:
        params.length_mm = 600.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.shape}_{args.installation}"
    png_path = save_static_plot(params, args.output_dir / f"{stem}.png")
    gif_path = None
    if not args.no_gif:
        gif_path = save_animation(
            params,
            args.output_dir / f"{stem}.gif",
            frame_count=args.frames,
            playback_speed=args.speed,
        )
    curve = calculate_curve(params)
    summary_path = args.output_dir / f"{stem}.json"
    summary_path.write_text(
        json.dumps(
            {"parameters": parameters_as_dict(params), "results": curve.summary()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"График: {png_path}")
    if gif_path:
        print(f"Анимация: {gif_path}")
    print(f"Результаты: {summary_path}")


if __name__ == "__main__":
    main()
