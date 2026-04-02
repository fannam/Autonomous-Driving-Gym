import sys
from pathlib import Path


def _load_main():
    if __package__:
        from .self_play_parallel import main as delegated_main

        return delegated_main

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from self_play_parallel import main as delegated_main

    return delegated_main


def main():
    print(
        "Warning: self_play_parallel_racetrack.py is deprecated; "
        "use self_play_parallel.py instead.",
        file=sys.stderr,
        flush=True,
    )
    return _load_main()()


if __name__ == "__main__":
    raise SystemExit(main())
