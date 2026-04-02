from __future__ import annotations
import argparse
import os
from typing import List, Optional
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from .common.hydra_config import resolve_config_dir as _resolve_config_dir
def _select_runner(task_name: str):
    if task_name == "dataset_register":
        from .processes.dataset_register import run
        return run
    if task_name == "preprocess":
        from .processes.preprocess import run
        return run
    if task_name == "train_model":
        from .processes.train_model import run
        return run
    if task_name == "train_ensemble":
        from .processes.train_ensemble import run
        return run
    if task_name == "leaderboard":
        from .processes.leaderboard import run
        return run
    if task_name == "infer":
        from .processes.infer import run
        return run
    if task_name == "pipeline":
        from .processes.pipeline import run
        return run
    if task_name == "retrain":
        from .processes.retrain import run
        return run
    raise ValueError(f"Unknown task.name: {task_name}")
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Compose config and print it, then exit.",
    )
    args, overrides = parser.parse_known_args(argv)
    config_dir = _resolve_config_dir(os.getenv("TABULAR_ANALYSIS_CONFIG_DIR"), __file__)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        return
    task_name = getattr(cfg.task, "name", None)
    if not task_name:
        raise ValueError("cfg.task.name is required (e.g. task=preprocess).")
    runner = _select_runner(task_name)
    runner(cfg)
if __name__ == "__main__":
    main()
