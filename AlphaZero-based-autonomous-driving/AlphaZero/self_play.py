try:
    from scripts.self_play import main, run_self_play
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from .scripts.self_play import main, run_self_play

__all__ = ["run_self_play", "main"]


if __name__ == "__main__":
    main()

