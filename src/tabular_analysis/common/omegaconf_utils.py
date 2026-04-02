from __future__ import annotations

from typing import Any

_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_OMEGACONF_RECOVERABLE_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)


def ensure_config_alias(cfg: Any, source_path: str, target_path: str) -> None:
    try:
        from omegaconf import OmegaConf
    except _OPTIONAL_IMPORT_ERRORS:
        return

    if not OmegaConf.is_config(cfg):
        return
    if OmegaConf.select(cfg, target_path) is not None:
        return

    source = OmegaConf.select(cfg, source_path)
    if source is None:
        return

    was_struct = False
    try:
        was_struct = OmegaConf.is_struct(cfg)
    except _OMEGACONF_RECOVERABLE_ERRORS:
        was_struct = False

    if was_struct:
        try:
            OmegaConf.set_struct(cfg, False)
        except _OMEGACONF_RECOVERABLE_ERRORS:
            pass

    try:
        OmegaConf.update(cfg, target_path, source, merge=False)
    finally:
        if was_struct:
            try:
                OmegaConf.set_struct(cfg, True)
            except _OMEGACONF_RECOVERABLE_ERRORS:
                pass

