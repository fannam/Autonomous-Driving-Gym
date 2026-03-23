#!/usr/bin/env python3
"""Bootstrap clone/install/run for parallel AlphaZero self-play."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/fannam/Autonomous-Driving-Gym.git"
SELF_PLAY_SCRIPT = Path(
    "AlphaZero-based-autonomous-driving/AlphaZero/scripts/self_play_parallel_racetrack.py"
)
DEFAULT_OUTPUT_DIR = Path(
    "AlphaZero-based-autonomous-driving/outputs/racetrack_self_play_parallel"
)
DEFAULT_ERROR_TAIL_LINES = 60


def log(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


def format_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def run_command(
    parts: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    quiet: bool = False,
    success_message: str | None = None,
) -> None:
    command_display = format_command(parts)
    if cwd is not None:
        log(f"$ {command_display}  (cwd={cwd})")
    else:
        log(f"$ {command_display}")
    if not quiet:
        subprocess.run(
            parts,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            check=True,
        )
        return

    completed = subprocess.run(
        parts,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        output = completed.stdout or ""
        tail = "\n".join(output.strip().splitlines()[-DEFAULT_ERROR_TAIL_LINES:])
        if tail:
            log("Command output tail:")
            print(tail, flush=True)
        raise subprocess.CalledProcessError(
            completed.returncode,
            parts,
            output=completed.stdout,
        )
    if success_message:
        log(success_message)


def is_repo_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").exists()
        and (path / "highway-env" / "pyproject.toml").exists()
        and (path / SELF_PLAY_SCRIPT).exists()
    )


def derive_repo_dir(repo_url: str, requested_repo_dir: str | None) -> Path:
    if requested_repo_dir:
        return Path(requested_repo_dir).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if is_repo_root(cwd):
        return cwd

    repo_name = repo_url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return (cwd / repo_name).resolve()


def clone_repo(
    repo_url: str,
    repo_dir: Path,
    *,
    branch: str | None,
    clone_depth: int | None,
    quiet: bool,
) -> Path:
    if is_repo_root(repo_dir):
        log(f"Reusing existing repository: {repo_dir}")
        return repo_dir

    if repo_dir.exists():
        raise RuntimeError(
            f"Target path already exists but is not a compatible repository: {repo_dir}"
        )

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone"]
    if branch:
        command.extend(["--branch", branch, "--single-branch"])
    if clone_depth is not None:
        if clone_depth <= 0:
            raise ValueError("--clone-depth must be a positive integer.")
        command.extend(["--depth", str(clone_depth)])
    command.extend([repo_url, str(repo_dir)])
    run_command(
        command,
        quiet=quiet,
        success_message=f"Repository cloned into: {repo_dir}",
    )
    return repo_dir


def choose_installer(installer: str) -> str:
    if installer == "auto":
        return "uv" if shutil.which("uv") else "pip"
    return installer


def has_option(args: list[str], option_name: str) -> bool:
    return any(arg == option_name or arg.startswith(f"{option_name}=") for arg in args)


def get_option_value(args: list[str], option_name: str) -> str | None:
    for index, arg in enumerate(args):
        if arg == option_name:
            next_index = index + 1
            if next_index < len(args):
                return args[next_index]
            return None
        if arg.startswith(f"{option_name}="):
            return arg.split("=", 1)[1]
    return None


def upsert_option(args: list[str], option_name: str, value: str) -> list[str]:
    updated_args = []
    skip_next = False
    replaced = False

    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == option_name:
            updated_args.extend([option_name, value])
            skip_next = True
            replaced = True
            continue
        if arg.startswith(f"{option_name}="):
            updated_args.extend([option_name, value])
            replaced = True
            continue
        updated_args.append(arg)

    if not replaced:
        updated_args.extend([option_name, value])
    return updated_args


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def install_with_uv(repo_dir: Path, *, quiet: bool) -> str:
    if shutil.which("uv") is None:
        raise RuntimeError("Installer 'uv' was requested, but the 'uv' binary was not found.")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    run_command(
        ["uv", "sync"],
        cwd=repo_dir,
        env=env,
        quiet=quiet,
        success_message="Dependencies installed with uv.",
    )
    return "uv"


def install_with_pip(repo_dir: Path, *, venv_dir: Path | None, quiet: bool) -> Path:
    target_venv_dir = (venv_dir or (repo_dir / ".venv")).resolve()
    python_bin = venv_python_path(target_venv_dir)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_PROGRESS_BAR"] = "off"

    if not python_bin.exists():
        run_command(
            [sys.executable, "-m", "venv", str(target_venv_dir)],
            cwd=repo_dir,
            env=env,
            quiet=quiet,
            success_message=f"Created virtual environment: {target_venv_dir}",
        )

    run_command(
        [str(python_bin), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        cwd=repo_dir,
        env=env,
        quiet=quiet,
        success_message="Upgraded pip in the virtual environment.",
    )
    run_command(
        [str(python_bin), "-m", "pip", "install", "--quiet", "-e", "highway-env"],
        cwd=repo_dir,
        env=env,
        quiet=quiet,
        success_message="Installed editable dependency: highway-env.",
    )
    run_command(
        [
            str(python_bin),
            "-m",
            "pip",
            "install",
            "--quiet",
            "-e",
            "AlphaZero-based-autonomous-driving",
        ],
        cwd=repo_dir,
        env=env,
        quiet=quiet,
        success_message="Installed editable package: AlphaZero-based-autonomous-driving.",
    )
    return python_bin


def build_runtime_env(matplotlib_backend: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = matplotlib_backend
    return env


def _probe_torch_cuda(
    runner_prefix: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> dict[str, object] | None:
    probe_code = """
import json
import warnings

warnings.filterwarnings("ignore")

result = {
    "cuda_available": False,
    "arch_list": [],
    "device_name": None,
    "device_capability": None,
    "torch_cuda_version": None,
}

try:
    import torch

    result["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    result["arch_list"] = list(getattr(torch.cuda, "get_arch_list", lambda: [])())
    result["cuda_available"] = bool(torch.cuda.is_available())
    if result["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        result["device_name"] = props.name
        result["device_capability"] = [int(props.major), int(props.minor)]
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(result))
""".strip()
    completed = subprocess.run(
        [*runner_prefix, "-c", probe_code],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        return None

    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _capability_tag(capability: object) -> str | None:
    if not isinstance(capability, list | tuple) or len(capability) != 2:
        return None
    major, minor = capability
    try:
        return f"sm_{int(major)}{int(minor)}"
    except (TypeError, ValueError):
        return None


def resolve_self_play_device(
    runner_prefix: list[str],
    repo_dir: Path,
    passthrough_args: list[str],
    runtime_env: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    requested_device = get_option_value(passthrough_args, "--device")
    if requested_device is not None and requested_device.lower() not in {"", "auto"}:
        return passthrough_args, runtime_env

    probe_env = dict(runtime_env)
    probe_env["PYTHONWARNINGS"] = "ignore"
    probe = _probe_torch_cuda(runner_prefix, cwd=repo_dir, env=probe_env)
    if not probe:
        log("Torch CUDA probe unavailable. Leaving self-play device unchanged.")
        return passthrough_args, runtime_env

    if not probe.get("cuda_available", False):
        return passthrough_args, runtime_env

    capability_tag = _capability_tag(probe.get("device_capability"))
    arch_list = [str(item) for item in probe.get("arch_list", [])]
    if capability_tag and arch_list and capability_tag not in arch_list:
        device_name = probe.get("device_name") or "unknown GPU"
        runtime_env = dict(runtime_env)
        runtime_env["CUDA_VISIBLE_DEVICES"] = ""
        log(
            "Installed PyTorch CUDA kernels do not support "
            f"{device_name} ({capability_tag}); forcing --device cpu."
        )
        filtered_args = upsert_option(passthrough_args, "--device", "cpu")
        return filtered_args, runtime_env

    return passthrough_args, runtime_env


def resolve_runner(
    repo_dir: Path,
    *,
    installer: str,
    skip_install: bool,
    venv_dir: Path | None,
    quiet_install: bool,
) -> tuple[list[str], str]:
    selected_installer = choose_installer(installer)

    if selected_installer == "uv":
        if not skip_install:
            try:
                install_with_uv(repo_dir, quiet=quiet_install)
            except subprocess.CalledProcessError:
                if installer != "auto":
                    raise
                log("uv installation failed in auto mode. Falling back to venv + pip.")
                python_bin = install_with_pip(
                    repo_dir,
                    venv_dir=venv_dir,
                    quiet=quiet_install,
                )
                return [str(python_bin), "-u"], str(python_bin)
        elif shutil.which("uv") is None:
            raise RuntimeError("Cannot skip install with installer 'uv' because 'uv' is not available.")
        return ["uv", "run", "python", "-u"], "uv"

    if selected_installer != "pip":
        raise RuntimeError(f"Unsupported installer: {selected_installer}")

    python_bin = venv_python_path((venv_dir or (repo_dir / ".venv")).resolve())
    if skip_install and not python_bin.exists():
        raise RuntimeError(
            f"Cannot skip install because the virtual environment does not exist yet: {python_bin}"
        )
    if not skip_install:
        python_bin = install_with_pip(
            repo_dir,
            venv_dir=venv_dir,
            quiet=quiet_install,
        )
    return [str(python_bin), "-u"], str(python_bin)


def build_self_play_command(runner_prefix: list[str], passthrough_args: list[str]) -> list[str]:
    command = list(runner_prefix)
    command.append(str(SELF_PLAY_SCRIPT))
    if passthrough_args:
        command.extend(passthrough_args)
    if not has_option(passthrough_args, "--output-dir"):
        command.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])
    return command


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Clone the Autonomous-Driving-Gym repository, install dependencies, "
            "and run the parallel AlphaZero racetrack self-play script."
        ),
        epilog=(
            "Any unknown arguments are forwarded to "
            "AlphaZero/scripts/self_play_parallel_racetrack.py.\n\n"
            "Example:\n"
            "  python bootstrap_parallel_self_play.py "
            "--repo-dir /tmp/adg-run --workers 4 --episodes-per-worker 8 --duration 120"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Git URL to clone.")
    parser.add_argument(
        "--repo-dir",
        default=None,
        help=(
            "Target directory for the repository. If omitted and the current working "
            "directory already looks like the repo root, that checkout is reused."
        ),
    )
    parser.add_argument("--branch", default=None, help="Optional branch or tag to clone.")
    parser.add_argument(
        "--clone-depth",
        type=int,
        default=1,
        help="Shallow clone depth. Use 0 or a negative value to disable --depth and perform a full clone.",
    )
    parser.add_argument(
        "--installer",
        choices=("auto", "uv", "pip"),
        default="auto",
        help="Dependency installer. 'auto' prefers uv and falls back to venv+pip.",
    )
    parser.add_argument(
        "--venv-dir",
        default=None,
        help="Custom virtualenv path used only when --installer=pip.",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip the clone step and expect --repo-dir to already point at the repo.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation and run with the existing environment.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Stop after clone/install without launching self-play.",
    )
    parser.add_argument(
        "--matplotlib-backend",
        default="Agg",
        help="Matplotlib backend forced for the self-play run. Use Agg on Kaggle/headless systems.",
    )
    parser.add_argument(
        "--verbose-install",
        action="store_true",
        help="Show full clone/install command output instead of the default compact summaries.",
    )
    args, passthrough_args = parser.parse_known_args()
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]
    return args, passthrough_args


def main() -> int:
    args, passthrough_args = parse_args()
    repo_dir = derive_repo_dir(args.repo_url, args.repo_dir)
    venv_dir = Path(args.venv_dir).expanduser().resolve() if args.venv_dir else None

    if args.skip_clone:
        if not is_repo_root(repo_dir):
            raise RuntimeError(
                f"--skip-clone was provided, but the repository checkout was not found at: {repo_dir}"
            )
        log(f"Skipping clone step and using repository: {repo_dir}")
    else:
        clone_depth = args.clone_depth if args.clone_depth > 0 else None
        clone_repo(
            args.repo_url,
            repo_dir,
            branch=args.branch,
            clone_depth=clone_depth,
            quiet=not args.verbose_install,
        )

    if not is_repo_root(repo_dir):
        raise RuntimeError(f"Repository layout is not valid for self-play: {repo_dir}")

    runner_prefix, runner_display = resolve_runner(
        repo_dir,
        installer=args.installer,
        skip_install=args.skip_install,
        venv_dir=venv_dir,
        quiet_install=not args.verbose_install,
    )
    log(f"Runner ready: {runner_display}")

    if args.skip_run:
        log("Skipping self-play execution.")
        return 0

    runtime_env = build_runtime_env(args.matplotlib_backend)
    passthrough_args, runtime_env = resolve_self_play_device(
        runner_prefix,
        repo_dir,
        passthrough_args,
        runtime_env,
    )
    self_play_command = build_self_play_command(runner_prefix, passthrough_args)
    log(
        "Runtime overrides: "
        f"PYTHONUNBUFFERED={runtime_env['PYTHONUNBUFFERED']} "
        f"MPLBACKEND={runtime_env['MPLBACKEND']}"
    )
    if runtime_env.get("CUDA_VISIBLE_DEVICES", None) == "":
        log("CUDA visibility override: hidden GPU devices for this run.")
    if not passthrough_args:
        log("No self-play overrides were provided. Using the defaults from the repo script.")
    run_command(self_play_command, cwd=repo_dir, env=runtime_env)
    log(f"Self-play outputs are expected under: {repo_dir / DEFAULT_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        log(f"Command failed with exit code {exc.returncode}: {format_command(exc.cmd)}")
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        log("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:  # pragma: no cover - CLI guard
        log(str(exc))
        raise SystemExit(1)
