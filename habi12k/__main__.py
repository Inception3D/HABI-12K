"""Command-line entry point for HABI-12K scene validation."""

import argparse
import sys

import av

from .scene import Scene


def main() -> int:
    parser = argparse.ArgumentParser(description="Work with packaged HABI-12K scenes.")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate a packaged scene.")
    validate.add_argument("scene", help="Directory containing transforms.json.")
    args = parser.parse_args()

    try:
        scene = Scene(args.scene)
        scene.validate()
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, av.FFmpegError) as error:
        print(f"{args.scene}: {error}", file=sys.stderr)
        return 1
    print(f"Validated {len(scene)} frames in {scene.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
