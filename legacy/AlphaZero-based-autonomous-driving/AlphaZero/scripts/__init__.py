try:
    from scripts.evaluate import main as evaluate_main
    from scripts.infer import main as infer_main
    from scripts.self_play import main as self_play_main
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from .evaluate import main as evaluate_main
    from .infer import main as infer_main
    from .self_play import main as self_play_main

__all__ = ["evaluate_main", "infer_main", "self_play_main"]
