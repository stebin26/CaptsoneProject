from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ops_common.config import settings
from ops_common.logging import configure_logging, get_logger
from app.generators import (
    INDUSTRY_SPECS,
    GeneratorConfig,
    generate_to_csv,
)

logger = get_logger(__name__)

_DEFAULT_OUT_DIR = Path("/data/samples")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-simulator",
        description="Generate realistic correlated multi-industry sample data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List available industries.")
    p_list.set_defaults(func=_cmd_list)

    p_gen = sub.add_parser("generate", help="Generate one industry's CSV.")
    p_gen.add_argument("industry", choices=sorted(INDUSTRY_SPECS.keys()))
    p_gen.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR))
    p_gen.add_argument("--days", type=int, default=90)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.set_defaults(func=_cmd_generate)

    p_all = sub.add_parser("generate-all", help="Generate CSVs for every industry.")
    p_all.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR))
    p_all.add_argument("--days", type=int, default=90)
    p_all.add_argument("--seed", type=int, default=42)
    p_all.set_defaults(func=_cmd_generate_all)

    p_seed = sub.add_parser(
        "seed", help="Generate all industries into the upload dir for demo."
    )
    p_seed.add_argument("--days", type=int, default=90)
    p_seed.add_argument("--seed", type=int, default=42)
    p_seed.set_defaults(func=_cmd_seed)

    return parser


def _cmd_list(args: argparse.Namespace) -> int:
    print("Available industries:\n")
    for key, spec in sorted(INDUSTRY_SPECS.items()):
        print(f"  {key:<15} {spec.entity_count:>3} entities  {spec.description}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    config = GeneratorConfig(days=args.days, seed=args.seed)
    path = generate_to_csv(args.industry, args.out_dir, config)
    print(f"Generated: {path}")
    return 0


def _cmd_generate_all(args: argparse.Namespace) -> int:
    config = GeneratorConfig(days=args.days, seed=args.seed)
    out_dir = Path(args.out_dir)
    written: list[Path] = []
    for industry in sorted(INDUSTRY_SPECS.keys()):
        path = generate_to_csv(industry, out_dir, config)
        written.append(path)
    print(f"Generated {len(written)} files in {out_dir}:")
    for p in written:
        print(f"  {p.name}")
    return 0


def _cmd_seed(args: argparse.Namespace) -> int:
    config = GeneratorConfig(days=args.days, seed=args.seed)
    out_dir = settings.upload_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for industry in sorted(INDUSTRY_SPECS.keys()):
        path = generate_to_csv(industry, out_dir, config)
        written.append(path)

    logger.info("Seeded sample data", extra={"count": len(written), "dir": str(out_dir)})
    print(f"Seeded {len(written)} sample CSVs into {out_dir}:")
    for p in written:
        print(f"  {p.name}")
    print("\nUpload any of these via the dashboard to run onboarding.")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Simulator command failed")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())