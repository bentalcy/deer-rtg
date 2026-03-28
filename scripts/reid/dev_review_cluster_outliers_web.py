import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import cast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dev runner with auto-reload for review_cluster_outliers_web.py"
    )
    _ = parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds for file changes.",
    )
    _ = parser.add_argument(
        "forward_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to review_cluster_outliers_web.py (prefix with --).",
    )
    return parser.parse_args()


def tracked_files(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "scripts" / "reid").glob("*.py"))


def snapshot_mtimes(paths: list[Path]) -> dict[Path, float]:
    out: dict[Path, float] = {}
    for path in paths:
        if path.exists():
            out[path] = path.stat().st_mtime
    return out


def has_changes(before: dict[Path, float], after: dict[Path, float]) -> bool:
    if before.keys() != after.keys():
        return True
    for path, mtime in before.items():
        if after.get(path) != mtime:
            return True
    return False


def pick_python(repo_root: Path) -> str:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def launch_child(python_bin: str, script_path: Path, forwarded: list[str]) -> subprocess.Popen[bytes]:
    command = [python_bin, str(script_path), *forwarded]
    return subprocess.Popen(command)


def normalize_forwarded(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main() -> int:
    args = parse_args()
    values = cast(dict[str, object], vars(args))

    interval_obj = values.get("interval")
    if not isinstance(interval_obj, (int, float)):
        raise ValueError("invalid --interval")
    interval = float(interval_obj)

    forward_obj = values.get("forward_args")
    if not isinstance(forward_obj, list):
        raise ValueError("invalid forwarded args")
    forwarded_input: list[str] = []
    for value in cast(list[object], forward_obj):
        if not isinstance(value, str):
            raise ValueError("invalid forwarded args")
        forwarded_input.append(value)

    repo_root = Path(__file__).resolve().parents[2]
    target_script = Path(__file__).resolve().with_name("review_cluster_outliers_web.py")
    forwarded = normalize_forwarded(forwarded_input)
    python_bin = pick_python(repo_root)

    watched = tracked_files(repo_root)
    previous = snapshot_mtimes(watched)
    child = launch_child(python_bin, target_script, forwarded)

    try:
        while True:
            return_code = child.poll()
            if return_code is not None:
                return return_code

            time.sleep(max(0.2, interval))
            watched = tracked_files(repo_root)
            current = snapshot_mtimes(watched)
            if not has_changes(previous, current):
                continue

            print("[dev-reload] Change detected. Restarting server...", flush=True)
            previous = current
            child.terminate()
            try:
                _ = child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                _ = child.wait(timeout=3)
            child = launch_child(python_bin, target_script, forwarded)
    except KeyboardInterrupt:
        print("\n[dev-reload] Stopping server.", flush=True)
        child.terminate()
        try:
            _ = child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            _ = child.wait(timeout=3)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
