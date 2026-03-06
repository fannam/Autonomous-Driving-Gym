try:
    from scripts.playground_test import main, run_action_playground
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from .scripts.playground_test import main, run_action_playground

__all__ = ["run_action_playground", "main"]


if __name__ == "__main__":
    main()
