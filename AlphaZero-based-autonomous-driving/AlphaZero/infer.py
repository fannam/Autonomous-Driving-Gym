try:
    from scripts.infer import main, run_inference
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from .scripts.infer import main, run_inference

__all__ = ["run_inference", "main"]


if __name__ == "__main__":
    main()

