import argparse

from .config import prepare_directories


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate red-wine quality models.")
    parser.add_argument("command", choices=["run", "clean", "eda", "split", "train", "evaluate", "predict"])
    parser.add_argument("--jobs", type=int, default=1, help="Parallel CV jobs (default: 1)")
    parser.add_argument("--input", help="Prediction CSV with the 11 required feature columns")
    parser.add_argument("--output", help="Prediction output CSV")
    args = parser.parse_args()
    prepare_directories()
    from .data import clean_data, split_data
    from .training import train_models
    from .evaluation import evaluate_models
    from .prediction import predict
    from .eda import run_eda
    steps = {"clean": clean_data, "eda": run_eda, "split": split_data,
             "train": lambda: train_models(args.jobs), "evaluate": evaluate_models,
             "predict": lambda: predict(args.input, args.output)}
    for name in steps if args.command == "run" else [args.command]:
        print(f"Running {name}...", flush=True)
        steps[name]()


if __name__ == "__main__":
    main()
