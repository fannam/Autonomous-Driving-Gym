try:
    from scripts.evaluate import evaluate_episode, main, run_batch_evaluation
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from .scripts.evaluate import evaluate_episode, main, run_batch_evaluation

__all__ = ["evaluate_episode", "run_batch_evaluation", "main"]


if __name__ == "__main__":
    main()
