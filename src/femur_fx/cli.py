"""Command-line entry point.

Examples::

    # one experiment
    python -m femur_fx run configs/example.yaml

    # every experiment in a folder, overriding the data location
    python -m femur_fx run configs/experiments --set data.image_dir=/path/to/images

    # noise characterisation of the images in data/ (paths from configs/data.yaml)
    python -m femur_fx noise

    # list the experiment configs that `run` would pick up
    python -m femur_fx list configs/experiments
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("femur_fx")


def _config_paths(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        p = Path(target)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml")))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"No such config file or folder: {target}")
    if not paths:
        raise FileNotFoundError(f"No .yaml configs found in {targets}")
    return paths


def cmd_run(args) -> int:
    from femur_fx.config import load_config
    from femur_fx.experiments import run_experiment

    # Validate every config before running anything.
    configs = [load_config(p, args.set) for p in _config_paths(args.configs)]
    if args.only:
        unknown = set(args.only) - {c.name for c in configs}
        if unknown:
            log.error("Unknown experiment name(s): %s", ", ".join(sorted(unknown)))
            return 1
        configs = [c for c in configs if c.name in set(args.only)]

    failures = []
    for cfg in configs:
        try:
            run_dir = run_experiment(cfg)
            log.info("Results for %s: %s", cfg.name, run_dir)
        except Exception:
            log.exception("Experiment %s failed", cfg.name)
            failures.append(cfg.name)
            if not args.keep_going:
                break
    if failures:
        log.error("Failed experiments: %s", ", ".join(failures))
        return 1
    return 0


def cmd_list(args) -> int:
    from femur_fx.config import load_config

    for path in _config_paths(args.configs):
        cfg = load_config(path)
        print(f"{cfg.name:<40} {cfg.model.type:<14} {path}")
        if cfg.description:
            print(f"    {cfg.description.strip()}")
    return 0


def cmd_noise(args) -> int:
    from femur_fx.analysis import run_noise_analysis
    from femur_fx.analysis.noise import load_variants
    from femur_fx.config import load_config

    # The config is only needed for the folders not given on the command line.
    cfg = None if (args.image_dir and args.out) else load_config(args.config)
    image_dir = args.image_dir or cfg.data.image_dir
    out = args.out or str(Path(cfg.output_dir) / "noise_analysis")
    variants = load_variants(args.variants) if args.variants else None
    run_noise_analysis(image_dir, out, args.example, variants)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="femur_fx", description="Femur fracture-type classification pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run experiments from YAML config files or folders")
    run.add_argument("configs", nargs="+", help="config files and/or folders of configs")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="override a config value, e.g. --set evaluation.n_splits=3 (repeatable)")
    run.add_argument("--only", nargs="+", metavar="NAME", help="run only experiments with these names")
    run.add_argument("--keep-going", action="store_true", help="continue with the next experiment after a failure")
    run.set_defaults(func=cmd_run)

    lst = sub.add_parser("list", help="list experiment configs")
    lst.add_argument("configs", nargs="+")
    lst.set_defaults(func=cmd_list)

    noise = sub.add_parser("noise", help="noise characterisation and denoising comparison")
    noise.add_argument("--config", default="configs/data.yaml",
                       help="config providing the default image folder and output folder")
    noise.add_argument("--image-dir", help="image folder (default: data.image_dir from --config)")
    noise.add_argument("--out", help="output folder (default: <output_dir>/noise_analysis)")
    noise.add_argument("--example", help="file name of the image to plot (default: first file)")
    noise.add_argument("--variants", help="YAML file with the filter settings to compare (default: a generic set)")
    noise.set_defaults(func=cmd_noise)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:  # setup problems, e.g. a missing or invalid config
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
