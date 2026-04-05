#!/usr/bin/env python3
"""Regression checks for ClearML runtime contracts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_PLATFORM_SRC = _REPO.parent / "ml_platform_v1-master" / "src"
for candidate in (_REPO, _SRC, _PLATFORM_SRC):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tabular_analysis.clearml import pipeline_templates as pipeline_template_module
from tabular_analysis.clearml import templates as template_module
from tabular_analysis.common.clearml_bootstrap import resolve_required_uv_extras
from tabular_analysis.common.clearml_config import read_clearml_api_section
from tabular_analysis.ops import clearml_identity as clearml_identity_module
from tabular_analysis import platform_adapter_task_ops as task_ops_module
from tabular_analysis.processes import pipeline as pipeline_module
from tabular_analysis.reporting import pipeline_report as pipeline_report_module
from tabular_analysis.processes.infer_support import resolve_batch_execution_mode
from tabular_analysis.processes.pipeline_support import (
    PIPELINE_RAW_DATASET_ID_SENTINEL,
    apply_pipeline_profile_defaults,
    build_pipeline_template_defaults,
    resolve_pipeline_profile,
    resolve_pipeline_run_flags,
    validate_pipeline_operator_inputs,
)
from tabular_analysis.registry.models import list_model_variants
from tools.clearml_templates import manage_templates as manage_templates_module
from tools.rehearsal import run_pipeline_v2 as rehearsal_module
from tools.clearml_entrypoint import (
    _extract_cli_keys,
    _normalize_loaded_override_key,
    _resolve_bootstrap_mode,
    _resolve_uv_settings,
    _store_loaded_override,
)
from omegaconf import OmegaConf


def _assert_clearml_hocon_reader() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "clearml.conf"
        path.write_text(
            'api {\n'
            '  api_server: "https://api.example.local"\n'
            '  web_server: "https://app.example.local"\n'
            '  files_server: "https://files.example.local"\n'
            '}\n',
            encoding="utf-8",
        )
        section = read_clearml_api_section(config_file=path)
        if section.get("api_server") != "https://api.example.local":
            raise AssertionError(f"unexpected api section: {section}")


def _assert_batch_execution_mode() -> None:
    inline = resolve_batch_execution_mode({"infer": {"batch": {"execution": "inline"}}}, clearml_enabled=False)
    if inline != "inline":
        raise AssertionError(f"unexpected inline execution: {inline}")
    children = resolve_batch_execution_mode({"infer": {"batch": {"execution": "clearml_children"}}}, clearml_enabled=True)
    if children != "clearml_children":
        raise AssertionError(f"unexpected child execution: {children}")
    try:
        resolve_batch_execution_mode({"infer": {"batch": {"execution": "clearml_children"}}}, clearml_enabled=False)
    except ValueError:
        return
    raise AssertionError("clearml_children should require ClearML")


def _assert_rehearsal_sync_cfg_keeps_clearml_enabled() -> None:
    cfg = rehearsal_module._build_minimal_clearml_cfg()
    clearml_cfg = getattr(getattr(cfg, "run", None), "clearml", None)
    if not bool(getattr(clearml_cfg, "enabled", False)):
        raise AssertionError(f"rehearsal sync cfg must keep ClearML enabled: {cfg}")
    if str(getattr(clearml_cfg, "execution", "local")) == "local":
        raise AssertionError(f"rehearsal sync cfg must not resolve to local execution: {cfg}")


def _assert_strict_template_lookup() -> None:
    calls: list[list[str]] = []
    original = {
        "list": template_module.list_clearml_tasks_by_tags,
        "status": template_module.clearml_task_status_from_obj,
        "tags": template_module.clearml_task_tags,
        "script": template_module.clearml_task_script,
        "id": template_module.clearml_task_id,
        "mismatch": template_module.clearml_script_mismatches,
        "spec": template_module.resolve_clearml_script_spec,
    }

    class _FakeTask:
        pass

    task = _FakeTask()

    try:
        template_module.list_clearml_tasks_by_tags = lambda tags, project_name=None: calls.append(list(tags)) or [task]
        template_module.clearml_task_status_from_obj = lambda task_obj: "created"
        template_module.clearml_task_tags = lambda task_obj: [
            "template:true",
            "task_kind:template",
            "usecase:TabularAnalysis",
            "process:infer",
            "schema:v1",
            "template_set:default",
            "solution:tabular-analysis",
        ]
        template_module.clearml_task_script = lambda task_obj: {"entry_point": "tools/clearml_entrypoint.py"}
        template_module.clearml_task_id = lambda task_obj: "template-123"
        template_module.clearml_script_mismatches = lambda expected, actual: False
        template_module.resolve_clearml_script_spec = lambda *args, **kwargs: {"entry_point": "tools/clearml_entrypoint.py"}
        task_id = template_module.resolve_template_task_id({"run": {"clearml": {"template_usecase_id": "TabularAnalysis", "template_set_id": "default"}}, "task": {"name": "infer"}}, "infer")
        if task_id != "template-123":
            raise AssertionError(f"unexpected task id: {task_id}")
        if len(calls) != 1:
            raise AssertionError(f"template lookup should be strict and single-pass: {calls}")
        tags = calls[0]
        if "template_set:default" not in tags or "schema:v1" not in tags:
            raise AssertionError(f"strict lookup missing canonical tags: {tags}")
    finally:
        template_module.list_clearml_tasks_by_tags = original["list"]
        template_module.clearml_task_status_from_obj = original["status"]
        template_module.clearml_task_tags = original["tags"]
        template_module.clearml_task_script = original["script"]
        template_module.clearml_task_id = original["id"]
        template_module.clearml_script_mismatches = original["mismatch"]
        template_module.resolve_clearml_script_spec = original["spec"]


def _assert_visible_pipeline_seed_lookup() -> None:
    calls: list[list[str]] = []
    original = {
        "list": pipeline_template_module.list_clearml_tasks_by_tags,
        "status": pipeline_template_module.clearml_task_status_from_obj,
        "tags": pipeline_template_module.clearml_task_tags,
        "script": pipeline_template_module.clearml_task_script,
        "id": pipeline_template_module.clearml_task_id,
        "runtime": pipeline_template_module._task_runtime,
        "artifacts": pipeline_template_module._task_artifact_names,
        "mismatch": pipeline_template_module.clearml_script_mismatches,
        "spec": pipeline_template_module.resolve_clearml_script_spec,
    }

    class _FakeTask:
        pass

    task = _FakeTask()

    try:
        pipeline_template_module.list_clearml_tasks_by_tags = lambda tags, project_name=None: calls.append(list(tags)) or [task]
        pipeline_template_module.clearml_task_status_from_obj = lambda task_obj: "completed"
        pipeline_template_module.clearml_task_tags = lambda task_obj: [
            "usecase:TabularAnalysis",
            "process:pipeline",
            "schema:v1",
            "template_set:default",
            "solution:tabular-analysis",
            "pipeline",
            "task_kind:seed",
            "pipeline_profile:pipeline",
        ]
        pipeline_template_module.clearml_task_script = lambda task_obj: {"entry_point": "tools/clearml_entrypoint.py"}
        pipeline_template_module.clearml_task_id = lambda task_obj: "pipeline-seed-123"
        pipeline_template_module._task_runtime = lambda task_obj: {"_pipeline_hash": "seed-hash"}
        pipeline_template_module._task_artifact_names = lambda task_obj: {
            "pipeline_run.json",
            "run_summary.json",
            "report.json",
            "manifest.json",
            "config_resolved.yaml",
        }
        pipeline_template_module.clearml_script_mismatches = lambda expected, actual: False
        pipeline_template_module.resolve_clearml_script_spec = lambda *args, **kwargs: {"entry_point": "tools/clearml_entrypoint.py"}
        task_id = pipeline_template_module.resolve_pipeline_seed_task_id(
            {
                "run": {
                    "clearml": {
                        "template_usecase_id": "TabularAnalysis",
                        "template_set_id": "default",
                    }
                }
            },
            pipeline_profile="pipeline",
        )
        if task_id != "pipeline-seed-123":
            raise AssertionError(f"unexpected pipeline seed id: {task_id}")
        if len(calls) != 1:
            raise AssertionError(f"pipeline seed lookup should be strict and single-pass: {calls}")
        tags = calls[0]
        for required in ("process:pipeline", "pipeline", "task_kind:seed", "pipeline_profile:pipeline", "template_set:default", "schema:v1"):
            if required not in tags:
                raise AssertionError(f"visible pipeline seed lookup missing {required}: {tags}")
    finally:
        pipeline_template_module.list_clearml_tasks_by_tags = original["list"]
        pipeline_template_module.clearml_task_status_from_obj = original["status"]
        pipeline_template_module.clearml_task_tags = original["tags"]
        pipeline_template_module.clearml_task_script = original["script"]
        pipeline_template_module.clearml_task_id = original["id"]
        pipeline_template_module._task_runtime = original["runtime"]
        pipeline_template_module._task_artifact_names = original["artifacts"]
        pipeline_template_module.clearml_script_mismatches = original["mismatch"]
        pipeline_template_module.resolve_clearml_script_spec = original["spec"]


def _assert_project_layout_contract() -> None:
    cfg = OmegaConf.create(
        {
            "run": {
                "clearml": {
                    "project_root": "LOCAL",
                    "template_usecase_id": "TabularAnalysis",
                    "template_set_id": "default",
                    "project_layout": {
                        "solution_root": "TabularAnalysis",
                        "pipeline_root_group": "Pipelines",
                        "pipeline_templates_group": "Templates",
                        "pipeline_runs_group": "Runs",
                        "templates_root_group": "Templates",
                        "step_templates_group": "Steps",
                        "runs_root_group": "Runs",
                        "separator": "/",
                        "group_map": {
                            "dataset_register": "01_Datasets",
                            "preprocess": "02_Preprocess",
                            "train_model": "03_TrainModels",
                            "train_ensemble": "04_Ensembles",
                            "infer": "05_Infer",
                            "infer_child": "05_Infer_Children",
                            "leaderboard": "99_Leaderboard",
                        },
                    },
                },
                "schema_version": "v1",
                "usecase_id": "demo_usecase",
            }
        }
    )
    context = clearml_identity_module.resolve_template_context(cfg)
    if clearml_identity_module.build_template_project_name(context, "preprocess", cfg=cfg) != "LOCAL/TabularAnalysis/Templates/Steps/02_Preprocess":
        raise AssertionError("step template project should resolve under Templates/Steps")
    if clearml_identity_module.build_project_name("LOCAL", "demo_usecase", "02_preprocess", process="preprocess", cfg=cfg) != "LOCAL/TabularAnalysis/Runs/demo_usecase/02_Preprocess":
        raise AssertionError("step run project should resolve under Runs/<usecase>")
    if clearml_identity_module.build_pipeline_template_project_name("LOCAL", pipeline_profile="pipeline", cfg=cfg) != "LOCAL/TabularAnalysis/.pipelines/pipeline":
        raise AssertionError("pipeline seed project should resolve under .pipelines/<profile>")
    if clearml_identity_module.build_pipeline_run_project_name("LOCAL", "demo_usecase", cfg=cfg) != "LOCAL/TabularAnalysis/Pipelines/Runs/demo_usecase":
        raise AssertionError("pipeline run project should resolve under Pipelines/Runs/<usecase>")


def _assert_runtime_tag_filter_contract() -> None:
    tags = clearml_identity_module.build_runtime_tags(
        process="pipeline",
        schema_version="v1",
        usecase_id="demo_usecase",
        pipeline_profile="pipeline",
        grid_run_id="grid123",
        extra_tags=[
            "template:true",
            "template_set:default",
            "usecase:TabularAnalysis",
            "process:pipeline",
            "solution:tabular-analysis",
            "custom:keep",
        ],
    )
    for blocked in ("template:true", "template_set:default", "usecase:TabularAnalysis"):
        if blocked in tags:
            raise AssertionError(f"runtime tags must drop template-only carryover: {tags}")
    for required in ("task_kind:run", "usecase:demo_usecase", "pipeline", "pipeline_profile:pipeline", "grid:grid123", "custom:keep"):
        if required not in tags:
            raise AssertionError(f"runtime tags missing {required}: {tags}")


def _assert_task_time_extras() -> None:
    if resolve_required_uv_extras(task_name="dataset_register") != []:
        raise AssertionError("dataset_register should not request optional extras")
    if resolve_required_uv_extras(task_name="preprocess") != []:
        raise AssertionError("preprocess should not request optional extras")
    if resolve_required_uv_extras(task_name="train_model", model_variant_name="lgbm") != ["lightgbm"]:
        raise AssertionError("lgbm train should use lightgbm extra only")
    if resolve_required_uv_extras(task_name="train_model", model_variant_name="xgboost") != ["xgboost"]:
        raise AssertionError("xgboost train should use xgboost extra only")
    if resolve_required_uv_extras(task_name="train_model", model_variant_name="catboost") != ["catboost"]:
        raise AssertionError("catboost train should use catboost extra only")
    if resolve_required_uv_extras(task_name="train_model", model_variant_name="ridge") != []:
        raise AssertionError("ridge train should not request optional extras")
    if resolve_required_uv_extras(task_name="infer", model_variant_name="lgbm") != ["lightgbm"]:
        raise AssertionError("lgbm infer should use lightgbm extra only")
    if resolve_required_uv_extras(task_name="infer", infer_mode="optimize") != ["models", "optuna"]:
        raise AssertionError("optimize infer should add optuna on top of the infer fallback")


def _assert_entrypoint_reads_clearml_slash_overrides() -> None:
    overrides = {
        "run/clearml/env/bootstrap": "uv",
        "run/clearml/env/uv/venv_dir": ".venv-test",
        "run/clearml/env/uv/frozen": "true",
        "task": "pipeline",
    }
    if _resolve_bootstrap_mode(overrides) != "uv":
        raise AssertionError("entrypoint must honor slash-form bootstrap overrides from ClearML task params")
    venv_dir, extras, all_extras, frozen = _resolve_uv_settings(overrides)
    if venv_dir != ".venv-test":
        raise AssertionError(f"unexpected slash-form venv_dir: {venv_dir}")
    if extras != []:
        raise AssertionError(f"pipeline task should not request optional extras: {extras}")
    if all_extras:
        raise AssertionError("slash-form all_extras should default to false")
    if not frozen:
        raise AssertionError("slash-form frozen should resolve to true")
    if _normalize_loaded_override_key("data/raw_dataset_id") != "data.raw_dataset_id":
        raise AssertionError("slash-form ClearML args must normalize to Hydra dot overrides")
    if _normalize_loaded_override_key("ops/clearml_policy") != "ops/clearml_policy":
        raise AssertionError("config-group style ops overrides must preserve slash form")
    if _normalize_loaded_override_key("run%2Egrid_run_id") != "run.grid_run_id":
        raise AssertionError("percent-encoded dotted ClearML args must normalize to Hydra dot overrides")
    if _normalize_loaded_override_key("group%2Fcustom") != "group/custom":
        raise AssertionError("percent-encoded config-group overrides must preserve slash form")
    loaded: dict[str, str] = {}
    priorities: dict[str, int] = {}
    _store_loaded_override(loaded, "data/raw_dataset_id", "REPLACE_WITH_EXISTING_RAW_DATASET_ID", priorities=priorities)
    _store_loaded_override(loaded, "data%2Eraw_dataset_id", "encoded_dataset", priorities=priorities)
    _store_loaded_override(loaded, "data.raw_dataset_id", "runtime_dataset", priorities=priorities)
    _store_loaded_override(loaded, "default_queue", "default", priorities=priorities)
    _store_loaded_override(loaded, "+pipeline.model_set", "regression_all", priorities=priorities)
    _store_loaded_override(loaded, "pipeline/model_set", "regression_all", priorities=priorities)
    _store_loaded_override(loaded, "pipeline/profile", "train_ensemble_full", priorities=priorities)
    if loaded.get("data.raw_dataset_id") != "runtime_dataset":
        raise AssertionError(f"dotted override must win over encoded/slash placeholders: {loaded}")
    if "default_queue" in loaded:
        raise AssertionError(f"controller-only default_queue should not be forwarded to Hydra: {loaded}")
    if loaded.get("pipeline.model_set") != "regression_all":
        raise AssertionError(f"pipeline.model_set must remain a plain override after config promotion: {loaded}")
    if "+pipeline.model_set" in loaded:
        raise AssertionError(f"pipeline.model_set must not be re-appended with + after config promotion: {loaded}")
    if loaded.get("+pipeline.profile") != "train_ensemble_full":
        raise AssertionError(f"runtime-only pipeline profile should be appended, not overridden: {loaded}")
    if _extract_cli_keys(["pipeline.model_set=regression_all", "task=pipeline"]) != {"pipeline.model_set", "task"}:
        raise AssertionError("CLI key extraction must treat +override and plain override as the same key")


def _assert_reset_clearml_task_args_replaces_stale_args() -> None:
    class _FakeTask:
        def __init__(self) -> None:
            self.params = {
                "Args/+pipeline.model_set": "regression_all",
                "Args/pipeline.model_set": "regression_all",
                "Args/run.usecase_id": "old",
                "General/name": "pipeline",
            }
            self.updated: dict[str, object] | None = None

        def set_parameters(self, payload: dict[str, object]) -> None:
            self.updated = dict(payload)
            self.params = dict(payload)

    fake = _FakeTask()
    originals = {
        "_get_clearml_task": task_ops_module._get_clearml_task,
        "_task_parameters": task_ops_module._task_parameters,
    }
    try:
        task_ops_module._get_clearml_task = lambda _task_id: fake
        task_ops_module._task_parameters = lambda _task: dict(fake.params)
        changed = task_ops_module.reset_clearml_task_args(
            "task-123",
            ["pipeline.model_set=regression_all", "run.usecase_id=new"],
        )
    finally:
        task_ops_module._get_clearml_task = originals["_get_clearml_task"]
        task_ops_module._task_parameters = originals["_task_parameters"]
    if not changed:
        raise AssertionError("reset_clearml_task_args should report changes for stale Args payloads")
    updated = fake.updated or {}
    if "Args/+pipeline.model_set" in updated:
        raise AssertionError(f"reset_clearml_task_args must remove stale appended keys: {updated}")
    if updated.get("Args/pipeline.model_set") != "regression_all":
        raise AssertionError(f"reset_clearml_task_args must keep plain pipeline.model_set only: {updated}")
    if updated.get("Args/run.usecase_id") != "new":
        raise AssertionError(f"reset_clearml_task_args must replace runtime args deterministically: {updated}")


def _assert_regression_model_set_contract() -> None:
    payload = OmegaConf.to_container(
        OmegaConf.load(_REPO / "conf" / "pipeline" / "model_sets" / "regression_all.yaml"),
        resolve=False,
    )
    if not isinstance(payload, dict):
        raise AssertionError("regression_all.yaml must be a mapping")
    variants = payload.get("variants")
    expected = [
        "catboost",
        "elasticnet",
        "extra_trees",
        "gaussian_process",
        "gradient_boosting",
        "knn",
        "lasso",
        "lgbm",
        "linear_regression",
        "mlp",
        "random_forest",
        "ridge",
        "xgboost",
    ]
    if variants != expected:
        raise AssertionError(f"unexpected regression_all variants: {variants}")
    regression_variants = list_model_variants("regression")
    classification_variants = list_model_variants("classification")
    if "svc" in regression_variants:
        raise AssertionError(f"svc must not be selectable for regression: {regression_variants}")
    if "svr" in regression_variants:
        raise AssertionError(f"svr must not be selectable for canonical regression lists: {regression_variants}")
    if "svc" not in classification_variants:
        raise AssertionError(f"svc must remain selectable for classification: {classification_variants}")
    if "svr" in classification_variants:
        raise AssertionError(f"svr must not be selectable for classification: {classification_variants}")


def _assert_explicit_pipeline_variants_override_model_set() -> None:
    cfg = OmegaConf.create(
        {
            "pipeline": {
                "model_set": "regression_all",
                "model_variants": ["ridge", "lgbm", "xgboost"],
                "grid": {"model_variants": ["ridge"]},
            }
        }
    )
    preprocess_variants, model_variants = pipeline_module._resolve_variants(cfg)
    if model_variants != ["ridge", "lgbm", "xgboost"]:
        raise AssertionError(f"explicit pipeline.model_variants must override model_set: {model_variants}")
    if preprocess_variants != []:
        raise AssertionError(f"unexpected preprocess variants in override regression: {preprocess_variants}")


def _assert_model_set_overrides_grid_default_variants() -> None:
    cfg = OmegaConf.create(
        {
            "pipeline": {
                "model_set": "regression_all",
                "grid": {"model_variants": ["ridge"]},
            }
        }
    )
    _, model_variants = pipeline_module._resolve_variants(cfg)
    expected = pipeline_module._resolve_model_set_variants("regression_all")
    if model_variants != expected:
        raise AssertionError(f"pipeline.model_set must win over default grid.model_variants for visible pipeline profiles: {model_variants}")


def _assert_pipeline_profile_defaults_clear_stale_model_variants() -> None:
    cfg = OmegaConf.create(
        {
            "pipeline": {
                "grid": {"model_variants": ["ridge"]},
                "model_variants": ["ridge"],
                "model_set": None,
            },
            "ensemble": {"enabled": False},
        }
    )
    updated = apply_pipeline_profile_defaults(cfg, "pipeline")
    preprocess_variants, model_variants = pipeline_module._resolve_variants(updated)
    if model_variants != [
        "catboost",
        "elasticnet",
        "extra_trees",
        "gaussian_process",
        "gradient_boosting",
        "knn",
        "lasso",
        "lgbm",
        "linear_regression",
        "mlp",
        "random_forest",
        "ridge",
        "xgboost",
    ]:
        raise AssertionError(f"pipeline profile defaults must expand canonical regression_all: {model_variants}")
    if preprocess_variants != []:
        raise AssertionError(f"unexpected preprocess variants after profile defaults: {preprocess_variants}")


def _assert_pipeline_profile_defaults_clear_stale_selection() -> None:
    cfg = OmegaConf.create(
        {
            "pipeline": {
                "selection": {
                    "enabled_preprocess_variants": ["legacy_preprocess"],
                    "enabled_model_variants": ["ridge"],
                }
            },
            "ensemble": {
                "enabled": True,
                "selection": {"enabled_methods": ["weighted"]},
            },
        }
    )
    updated = apply_pipeline_profile_defaults(cfg, "train_ensemble_full")
    if list(updated.pipeline.selection.enabled_preprocess_variants) != []:
        raise AssertionError(f"pipeline profile defaults must clear stale preprocess selection: {updated}")
    if list(updated.pipeline.selection.enabled_model_variants) != []:
        raise AssertionError(f"pipeline profile defaults must clear stale model selection: {updated}")
    if list(updated.ensemble.methods) != ["mean_topk", "weighted", "stacking"]:
        raise AssertionError(f"train_ensemble_full must restore the canonical ensemble method mother set: {updated}")
    if list(updated.ensemble.selection.enabled_methods) != []:
        raise AssertionError(f"pipeline profile defaults must clear stale ensemble selection: {updated}")


def _assert_pipeline_profile_sets_run_flags() -> None:
    cfg = OmegaConf.create(
        {
            "pipeline": {
                "profile": "train_ensemble_full",
            },
            "ensemble": {"enabled": False},
        }
    )
    flags = resolve_pipeline_run_flags(cfg)
    expected = {
        "run_dataset_register": False,
        "run_preprocess": True,
        "run_train": True,
        "run_train_ensemble": True,
        "run_leaderboard": True,
        "run_infer": False,
    }
    if flags != expected:
        raise AssertionError(f"pipeline profile must supply the canonical run flags: {flags}")
    methods = pipeline_module._resolve_ensemble_methods(cfg)
    if methods != ["mean_topk", "weighted", "stacking"]:
        raise AssertionError(f"train_ensemble_full profile must provide the canonical ensemble methods: {methods}")


def _assert_pipeline_controller_context_attaches_by_task_id() -> None:
    class _FakeTaskObject:
        name = "pipeline"

        def get_project_name(self) -> str:
            return "LOCAL/TabularAnalysis/Pipelines"

    class _FakeTaskApi:
        @staticmethod
        def current_task() -> None:
            return None

        @staticmethod
        def get_task(task_id: str | None = None) -> _FakeTaskObject | None:
            if task_id != "fake-controller-task":
                raise AssertionError(f"unexpected task lookup: {task_id}")
            return fake_task

    fake_task = _FakeTaskObject()
    original_module = sys.modules.get("clearml")
    original_task_id = os.environ.get("CLEARML_TASK_ID")
    sys.modules["clearml"] = type("_FakeClearMLModule", (), {"Task": _FakeTaskApi})()
    os.environ["CLEARML_TASK_ID"] = "fake-controller-task"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create(
                {
                    "task": {"stage": "99_pipeline"},
                    "run": {
                        "output_dir": tmpdir,
                        "usecase_id": "fake-usecase",
                        "clearml": {"project_name": "LOCAL/TabularAnalysis/Pipelines"},
                    },
                }
            )
            ctx = pipeline_module._create_pipeline_controller_runtime_context(cfg)
            if ctx.task is not fake_task:
                raise AssertionError("pipeline controller runtime context must attach to CLEARML_TASK_ID when current_task() is None")
    finally:
        if original_module is None:
            sys.modules.pop("clearml", None)
        else:
            sys.modules["clearml"] = original_module
        if original_task_id is None:
            os.environ.pop("CLEARML_TASK_ID", None)
        else:
            os.environ["CLEARML_TASK_ID"] = original_task_id


def _assert_pipeline_controller_context_attaches_by_pipeline_task_id_override() -> None:
    class _FakeTaskObject:
        name = "pipeline"

        def get_project_name(self) -> str:
            return "LOCAL/TabularAnalysis/Pipelines"

    class _FakeTaskApi:
        @staticmethod
        def current_task() -> None:
            return None

        @staticmethod
        def get_task(task_id: str | None = None) -> _FakeTaskObject | None:
            if task_id != "fake-controller-task":
                raise AssertionError(f"unexpected task lookup: {task_id}")
            return fake_task

    fake_task = _FakeTaskObject()
    original_module = sys.modules.get("clearml")
    original_task_id = os.environ.get("CLEARML_TASK_ID")
    original_trains_task_id = os.environ.get("TRAINS_TASK_ID")
    if "CLEARML_TASK_ID" in os.environ:
        os.environ.pop("CLEARML_TASK_ID")
    if "TRAINS_TASK_ID" in os.environ:
        os.environ.pop("TRAINS_TASK_ID")
    sys.modules["clearml"] = type("_FakeClearMLModule", (), {"Task": _FakeTaskApi})()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create(
                {
                    "task": {"stage": "99_pipeline"},
                    "run": {
                        "output_dir": tmpdir,
                        "usecase_id": "fake-usecase",
                        "clearml": {
                            "project_name": "LOCAL/TabularAnalysis/Pipelines",
                            "pipeline_task_id": "fake-controller-task",
                        },
                    },
                }
            )
            ctx = pipeline_module._create_pipeline_controller_runtime_context(cfg)
            if ctx.task is not fake_task:
                raise AssertionError(
                    "pipeline controller runtime context must attach to run.clearml.pipeline_task_id when env lookup is unavailable"
                )
    finally:
        if original_module is None:
            sys.modules.pop("clearml", None)
        else:
            sys.modules["clearml"] = original_module
        if original_task_id is None:
            os.environ.pop("CLEARML_TASK_ID", None)
        else:
            os.environ["CLEARML_TASK_ID"] = original_task_id
        if original_trains_task_id is None:
            os.environ.pop("TRAINS_TASK_ID", None)
        else:
            os.environ["TRAINS_TASK_ID"] = original_trains_task_id


def _assert_rehearsal_rebuild_pipeline_outputs_from_clearml() -> None:
    class _FakeTask:
        def __init__(self, task_id: str, project: str, status: str, tags: list[str]) -> None:
            self.id = task_id
            self.project = project
            self.status = status
            self.tags = tags

    import tabular_analysis.platform_adapter_task as platform_task_module

    original_platform = {
        "list": platform_task_module.list_clearml_tasks_by_tags,
        "id": platform_task_module.clearml_task_id,
        "project": platform_task_module.clearml_task_project_name,
        "status": platform_task_module.clearml_task_status_from_obj,
        "tags": platform_task_module.clearml_task_tags,
    }
    original_get_status = pipeline_module.get_clearml_task_status
    original_report_bundle = pipeline_report_module.build_pipeline_report_bundle
    try:
        fake_tasks = [
            _FakeTask(
                "pipeline-task",
                "LOCAL/TabularAnalysis/Pipelines/Runs/demo_usecase",
                "completed",
                [
                    "solution:tabular-analysis",
                    "process:pipeline",
                    "task_kind:run",
                    "usecase:demo_usecase",
                    "grid:grid123",
                    "pipeline_profile:pipeline",
                ],
            ),
            _FakeTask(
                "preprocess-task",
                "LOCAL/TabularAnalysis/Runs/demo_usecase/02_Preprocess",
                "completed",
                [
                    "solution:tabular-analysis",
                    "process:preprocess",
                    "usecase:demo_usecase",
                    "preprocess:stdscaler_ohe",
                ],
            ),
            _FakeTask(
                "ridge-task",
                "LOCAL/TabularAnalysis/Runs/demo_usecase/03_TrainModels",
                "completed",
                [
                    "solution:tabular-analysis",
                    "process:train_model",
                    "usecase:demo_usecase",
                    "preprocess:stdscaler_ohe",
                    "model:ridge",
                ],
            ),
            _FakeTask(
                "lgbm-task",
                "LOCAL/TabularAnalysis/Runs/demo_usecase/03_TrainModels",
                "completed",
                [
                    "solution:tabular-analysis",
                    "process:train_model",
                    "usecase:demo_usecase",
                    "preprocess:stdscaler_ohe",
                    "model:lgbm",
                ],
            ),
            _FakeTask(
                "leaderboard-task",
                "LOCAL/TabularAnalysis/Runs/demo_usecase/99_Leaderboard",
                "completed",
                [
                    "solution:tabular-analysis",
                    "process:leaderboard",
                    "usecase:demo_usecase",
                ],
            ),
        ]

        platform_task_module.list_clearml_tasks_by_tags = lambda tags, project_name=None, task_name=None, allow_archived=True, order_by=None: fake_tasks
        platform_task_module.clearml_task_id = lambda task: task.id
        platform_task_module.clearml_task_project_name = lambda task: task.project
        platform_task_module.clearml_task_status_from_obj = lambda task: task.status
        platform_task_module.clearml_task_tags = lambda task: list(task.tags)
        pipeline_module.get_clearml_task_status = lambda task_id: {
            "ridge-task": "completed",
            "lgbm-task": "completed",
        }.get(task_id, "completed")

        def _fake_report_bundle(pipeline_run, *, cfg=None, max_models=5, pipeline_run_dir=None, pipeline_task_id=None):
            return pipeline_report_module.PipelineReportBundle(
                markdown="# Pipeline Summary\n",
                payload={"status": pipeline_run.get("status"), "summary": {"completed_jobs": pipeline_run.get("completed_jobs")}},
                links={"pipeline": {"task_id": pipeline_task_id}},
            )

        pipeline_report_module.build_pipeline_report_bundle = _fake_report_bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            stage_dir = output_dir / "99_pipeline"
            stage_dir.mkdir(parents=True, exist_ok=True)
            summary = rehearsal_module._rebuild_pipeline_outputs_from_clearml(
                repo=_REPO,
                output_dir=output_dir,
                usecase_id="demo_usecase",
                pipeline_task_id="pipeline-task",
                controller_status="completed",
                project_root="LOCAL",
            )
            if summary.get("status") != "completed":
                raise AssertionError(f"rebuild should finalize completed status: {summary}")
            if summary.get("planned_jobs") != 2:
                raise AssertionError(f"rebuild should reconstruct planned_jobs from ClearML child tasks: {summary}")
            if summary.get("completed_jobs") != 2:
                raise AssertionError(f"rebuild should count completed train jobs: {summary}")
            if len(summary.get("train_refs") or []) != 2:
                raise AssertionError(f"rebuild should repopulate train refs: {summary}")
            if summary.get("grid_run_id") != "grid123":
                raise AssertionError(f"rebuild should recover grid_run_id from the controller tags: {summary}")
            report_payload = json.loads((stage_dir / "report.json").read_text(encoding="utf-8"))
            if report_payload.get("status") != "completed":
                raise AssertionError(f"report.json should reflect rebuilt completed status: {report_payload}")
            rebuilt_pipeline_run = json.loads((stage_dir / "pipeline_run.json").read_text(encoding="utf-8"))
            if rebuilt_pipeline_run.get("completed_jobs") != 2:
                raise AssertionError(
                    f"rebuild should write pipeline_run.json from live ClearML state when synced artifacts are absent: {rebuilt_pipeline_run}"
                )
    finally:
        platform_task_module.list_clearml_tasks_by_tags = original_platform["list"]
        platform_task_module.clearml_task_id = original_platform["id"]
        platform_task_module.clearml_task_project_name = original_platform["project"]
        platform_task_module.clearml_task_status_from_obj = original_platform["status"]
        platform_task_module.clearml_task_tags = original_platform["tags"]
        pipeline_module.get_clearml_task_status = original_get_status
        pipeline_report_module.build_pipeline_report_bundle = original_report_bundle


def _assert_pipeline_step_references_are_not_quoted() -> None:
    payload = pipeline_module._build_pipeline_step_parameter_override_payload(
        {
            "data.raw_dataset_id": "${pipeline.data/raw_dataset_id}",
            "leaderboard.train_task_ids": ["${train__a.id}", "${train__b.id}"],
            "group/preprocess": "stdscaler_ohe",
        }
    )
    if payload.get("Args/data.raw_dataset_id") != "${pipeline.data/raw_dataset_id}":
        raise AssertionError(f"pipeline parameter reference must remain raw: {payload}")
    if payload.get("Args/leaderboard.train_task_ids") != ["${train__a.id}", "${train__b.id}"]:
        raise AssertionError(f"step reference lists must remain raw: {payload}")
    if payload.get("Args/group/preprocess") != "stdscaler_ohe":
        raise AssertionError(f"plain scalar overrides must remain unchanged: {payload}")


def _assert_pipeline_steps_enable_recursive_parameter_parsing() -> None:
    captured: dict[str, object] = {}

    class _FakeController:
        def add_step(
            self,
            *,
            name: str,
            base_task_id: str,
            parents: list[str] | None=None,
            parameter_override: dict[str, object] | None=None,
            recursively_parse_parameters: bool=False,
            execution_queue: str | None=None,
            clone_base_task: bool=True,
            cache_executed_step: bool=False,
        ) -> None:
            captured.update(
                {
                    "name": name,
                    "base_task_id": base_task_id,
                    "parents": parents,
                    "parameter_override": parameter_override,
                    "recursively_parse_parameters": recursively_parse_parameters,
                    "execution_queue": execution_queue,
                    "clone_base_task": clone_base_task,
                    "cache_executed_step": cache_executed_step,
                }
            )

    pipeline_module._add_pipeline_step(
        _FakeController(),
        name="leaderboard",
        base_task_id="template-123",
        parents=["train__a", "train__b"],
        parameter_override={
            "Args/leaderboard.train_task_ids": ["${train__a.id}", "${train__b.id}"],
        },
        execution_queue="default",
        clone_base_task=True,
        cache_executed_step=False,
    )
    if captured.get("recursively_parse_parameters") is not True:
        raise AssertionError(f"pipeline steps must enable recursive parameter parsing for list step refs: {captured}")


def _assert_leaderboard_bootstrap_extras_follow_pipeline_model_variants() -> None:
    cfg = OmegaConf.create(
        {
            "task": {"stage": "99_pipeline"},
            "pipeline": {
                "preprocess_variant": "stdscaler_ohe",
                "model_set": "regression_all",
                "model_variants": ["ridge", "lgbm", "xgboost"],
                "grid": {"model_variants": ["ridge"], "max_jobs": 50},
                "run_dataset_register": False,
                "run_preprocess": True,
                "run_train": True,
                "run_train_ensemble": False,
                "run_leaderboard": True,
                "run_infer": False,
                "plan_only": False,
                "hpo": {"enabled": False, "params": {}},
            },
            "run": {
                "output_dir": "outputs",
                "usecase_id": "demo_usecase",
                "schema_version": "v1",
                "clearml": {
                    "enabled": True,
                    "execution": "pipeline_controller",
                    "project_root": "LOCAL",
                    "queue_name": "default",
                    "env": {"bootstrap": "uv", "uv": {"venv_dir": ".venv", "frozen": True, "all_extras": False}},
                },
            },
            "exec_policy": {
                "queues": {
                    "default": "default",
                    "pipeline": "controller",
                    "train_model_heavy": "heavy-model",
                    "heavy_model_variants": ["catboost", "xgboost"],
                },
                "limits": {"max_jobs": 50, "max_models": 10, "max_hpo_trials": 0},
            },
            "eval": {
                "primary_metric": "rmse",
                "direction": "auto",
                "cv_folds": 5,
                "seed": 42,
                "task_type": "regression",
                "classification": {"mode": "auto", "top_k": 1},
                "metrics": {"classification_multiclass": ["accuracy", "f1_macro", "logloss"]},
                "calibration": {"enabled": False},
                "uncertainty": {"enabled": False},
                "ci": {"enabled": False},
            },
            "leaderboard": {"top_k": 10},
            "data": {"raw_dataset_id": "raw-123", "target_column": "target", "split": {"strategy": "random", "test_size": 0.2, "seed": 42}},
        }
    )
    plan = pipeline_module._build_pipeline_plan(cfg, "grid-demo", child_execution="logging")
    leaderboard_step = plan["steps"]["leaderboard"]
    if leaderboard_step is None:
        raise AssertionError("leaderboard step must exist in pipeline plan")
    extras = leaderboard_step["overrides"].get("run.clearml.env.uv.extras")
    if extras != ["lightgbm", "xgboost"]:
        raise AssertionError(f"leaderboard bootstrap extras must follow planned optional model variants: {extras}")


def _assert_loaded_pipeline_controller_reseeds_runtime_defaults() -> None:
    class _FakeTask:
        pass

    class _FakeController:
        def __init__(self) -> None:
            self._nodes = {"preprocess": object()}
            self.started_with: str | None = "unset"
            self.wait_called = False

        def start_locally(self, *, run_pipeline_steps_locally: bool = False) -> None:
            self.started_with = "local" if not run_pipeline_steps_locally else "fully-local"

        def wait(self, timeout: float | None = None) -> bool:
            self.wait_called = True
            return True

    fake_controller = _FakeController()
    fake_task = _FakeTask()
    originals = {
        "resolve_contract": pipeline_module._resolve_visible_pipeline_run_contract,
        "clearml_task_id": pipeline_module.clearml_task_id,
        "apply_defaults": pipeline_module._apply_visible_pipeline_run_defaults,
        "load_controller": pipeline_module.load_pipeline_controller_from_task,
        "add_steps": pipeline_module._add_clearml_pipeline_steps,
        "collect_step_ids": pipeline_module._collect_step_task_ids,
        "build_refs": pipeline_module._build_clearml_pipeline_refs,
        "build_summary": pipeline_module._build_local_pipeline_run_summary,
    }
    try:
        pipeline_module._resolve_visible_pipeline_run_contract = lambda **kwargs: pipeline_module._VisiblePipelineRunContract(
            plan={
                "queues": {"default": "default"},
                "steps": {"train": [], "preprocess": [], "train_ensemble": [], "dataset_register": None, "leaderboard": None, "infer": None},
                "plan_only": False,
                "run_overrides": {},
                "data_overrides": {},
                "downstream_data_overrides": {},
                "eval_overrides": {},
                "run_dataset_register": False,
                "run_preprocess": True,
                "run_train": False,
                "run_train_ensemble": False,
                "run_leaderboard": False,
                "run_infer": False,
                "preprocess_variants": ["stdscaler_ohe"],
                "model_variants": [],
            },
            pipeline_profile="train_ensemble_full",
            metadata={},
            queue_name="controller",
        )
        pipeline_module.clearml_task_id = lambda task: "controller-123"
        pipeline_module._apply_visible_pipeline_run_defaults = lambda **kwargs: {"run.grid_run_id": "grid-001"}
        pipeline_module.load_pipeline_controller_from_task = lambda source_task: fake_controller
        pipeline_module._add_clearml_pipeline_steps = lambda **kwargs: setattr(
            fake_controller,
            "_nodes",
            {
                "preprocess__stdscaler_ohe": object(),
                "train__stdscaler_ohe__ridge": object(),
                "leaderboard": object(),
            },
        )
        pipeline_module._collect_step_task_ids = lambda controller: {}
        pipeline_module._build_clearml_pipeline_refs = lambda **kwargs: (None, [], [], [], None, None)
        pipeline_module._build_local_pipeline_run_summary = lambda **kwargs: {"status": "stub"}
        cfg = OmegaConf.create({"run": {"clearml": {"queue_name": "default"}}})
        ctx = pipeline_module.TaskContext(task=fake_task, project_name="LOCAL/TabularAnalysis/Pipelines", task_name="pipeline", output_dir=Path.cwd())
        summary = pipeline_module._execute_current_pipeline_controller(cfg=cfg, ctx=ctx, grid_run_id="grid-001")
        if fake_controller.started_with != "local":
            raise AssertionError(f"remote controller should run controller logic in-place: {fake_controller.started_with}")
        expected_nodes = {"preprocess__stdscaler_ohe", "train__stdscaler_ohe__ridge", "leaderboard"}
        if set((getattr(fake_controller, "_nodes", {}) or {}).keys()) != expected_nodes:
            raise AssertionError(f"loaded controller should rebuild runtime nodes from the current plan: {getattr(fake_controller, '_nodes', None)}")
        if getattr(fake_controller, "_default_execution_queue", None) != "default":
            raise AssertionError(f"loaded controller should reseed default execution queue: {getattr(fake_controller, '_default_execution_queue', None)}")
        pipeline_args = getattr(fake_controller, "_pipeline_args", {})
        if pipeline_args.get("run/grid_run_id") != "grid-001":
            raise AssertionError(f"loaded controller should reseed pipeline args: {pipeline_args}")
        if not getattr(fake_controller, "wait_called", False):
            raise AssertionError("loaded controller should wait for child task completion before summarizing")
        if summary.get("pipeline_task_id") != "controller-123":
            raise AssertionError(f"unexpected pipeline summary: {summary}")
    finally:
        pipeline_module._resolve_visible_pipeline_run_contract = originals["resolve_contract"]
        pipeline_module.clearml_task_id = originals["clearml_task_id"]
        pipeline_module._apply_visible_pipeline_run_defaults = originals["apply_defaults"]
        pipeline_module.load_pipeline_controller_from_task = originals["load_controller"]
        pipeline_module._add_clearml_pipeline_steps = originals["add_steps"]
        pipeline_module._collect_step_task_ids = originals["collect_step_ids"]
        pipeline_module._build_clearml_pipeline_refs = originals["build_refs"]
        pipeline_module._build_local_pipeline_run_summary = originals["build_summary"]


def _assert_plan_only_controller_does_not_launch_steps() -> None:
    class _FakeTask:
        pass

    class _FakeController:
        def __init__(self) -> None:
            self._nodes = {"preprocess": object()}
            self.started = False
            self.serialized = False

        def start_locally(self, *, run_pipeline_steps_locally: bool = False) -> None:
            self.started = True

        def _serialize_pipeline_task(self) -> None:
            self.serialized = True

    fake_controller = _FakeController()
    fake_task = _FakeTask()
    originals = {
        "resolve_contract": pipeline_module._resolve_visible_pipeline_run_contract,
        "clearml_task_id": pipeline_module.clearml_task_id,
        "apply_defaults": pipeline_module._apply_visible_pipeline_run_defaults,
        "load_controller": pipeline_module.load_pipeline_controller_from_task,
        "add_steps": pipeline_module._add_clearml_pipeline_steps,
        "build_summary": pipeline_module._build_local_pipeline_run_summary,
    }
    try:
        pipeline_module._resolve_visible_pipeline_run_contract = lambda **kwargs: pipeline_module._VisiblePipelineRunContract(
            plan={
                "queues": {"default": "default"},
                "steps": {"train": [], "preprocess": [], "train_ensemble": [], "dataset_register": None, "leaderboard": None, "infer": None},
                "plan_only": True,
                "run_overrides": {},
                "data_overrides": {},
                "downstream_data_overrides": {},
                "eval_overrides": {},
                "run_dataset_register": False,
                "run_preprocess": True,
                "run_train": False,
                "run_train_ensemble": False,
                "run_leaderboard": False,
                "run_infer": False,
                "preprocess_variants": ["stdscaler_ohe"],
                "model_variants": [],
            },
            pipeline_profile="pipeline",
            metadata={},
            queue_name="controller",
        )
        pipeline_module.clearml_task_id = lambda task: "controller-plan-only"
        pipeline_module._apply_visible_pipeline_run_defaults = lambda **kwargs: {"run.grid_run_id": "grid-plan-only"}
        pipeline_module.load_pipeline_controller_from_task = lambda source_task: fake_controller
        pipeline_module._add_clearml_pipeline_steps = lambda **kwargs: setattr(fake_controller, "_nodes", {"preprocess__stdscaler_ohe": object()})
        pipeline_module._build_local_pipeline_run_summary = lambda **kwargs: {"status": "stub", "plan_only": kwargs["plan"]["plan_only"]}
        cfg = OmegaConf.create({"run": {"clearml": {"queue_name": "default"}}})
        ctx = pipeline_module.TaskContext(task=fake_task, project_name="LOCAL/TabularAnalysis/Pipelines", task_name="pipeline", output_dir=Path.cwd())
        summary = pipeline_module._execute_current_pipeline_controller(cfg=cfg, ctx=ctx, grid_run_id="grid-plan-only")
        if fake_controller.started:
            raise AssertionError("plan_only controller should not launch pipeline steps")
        if not fake_controller.serialized:
            raise AssertionError("plan_only controller should serialize the pipeline graph for UI visibility")
        if summary.get("status") != "planned":
            raise AssertionError(f"plan_only controller should return planned summary: {summary}")
        if summary.get("pipeline_task_id") != "controller-plan-only":
            raise AssertionError(f"plan_only summary missing controller task id: {summary}")
    finally:
        pipeline_module._resolve_visible_pipeline_run_contract = originals["resolve_contract"]
        pipeline_module.clearml_task_id = originals["clearml_task_id"]
        pipeline_module._apply_visible_pipeline_run_defaults = originals["apply_defaults"]
        pipeline_module.load_pipeline_controller_from_task = originals["load_controller"]
        pipeline_module._add_clearml_pipeline_steps = originals["add_steps"]
        pipeline_module._build_local_pipeline_run_summary = originals["build_summary"]


def _assert_pipeline_template_defaults_keep_plan_only() -> None:
    cfg = OmegaConf.create(
        {
            "pipeline": {"model_set": "regression_all"},
            "run": {"clearml": {"queue_name": "default"}},
        }
    )
    defaults = build_pipeline_template_defaults(
        cfg=cfg,
        plan={
            "run_overrides": {},
            "data_overrides": {},
            "downstream_data_overrides": {},
            "eval_overrides": {},
            "queues": {"default": "default"},
            "run_dataset_register": False,
            "run_preprocess": True,
            "run_train": True,
            "run_train_ensemble": False,
            "run_leaderboard": True,
            "run_infer": False,
            "plan_only": True,
            "preprocess_variants": ["stdscaler_ohe"],
            "model_variants": ["ridge"],
            "selection": {
                "requested_preprocess_variants": ["stdscaler_ohe"],
                "active_preprocess_variants": ["stdscaler_ohe"],
                "disabled_preprocess_variants": [],
                "requested_model_variants": ["ridge", "lgbm"],
                "active_model_variants": ["ridge"],
                "disabled_model_variants": ["lgbm"],
                "requested_ensemble_methods": [],
                "active_ensemble_methods": [],
                "disabled_ensemble_methods": [],
            },
        },
        grid_run_id="grid-plan-only",
        pipeline_profile="pipeline",
        pipeline_task_id="controller-plan-only",
    )
    if defaults.get("pipeline.plan_only") is not True:
        raise AssertionError(f"pipeline seed defaults must preserve plan_only: {defaults}")
    if defaults.get("pipeline.grid.model_variants") != ["ridge", "lgbm"]:
        raise AssertionError(f"pipeline seed defaults must keep requested model superset in grid.model_variants: {defaults}")
    if defaults.get("pipeline.selection.enabled_model_variants") != ["ridge"]:
        raise AssertionError(f"pipeline seed defaults must expose active model subset via selection: {defaults}")


def _assert_ui_clone_cfg_normalization_clears_seed_only_defaults() -> None:
    cfg = OmegaConf.create(
        {
            "data": {"raw_dataset_id": "dataset-123"},
            "pipeline": {"plan_only": True, "dry_run": True, "plan": True},
            "run": {"grid_run_id": "seed__train_ensemble_full"},
        }
    )
    pipeline_module._normalize_ui_cloned_pipeline_cfg(cfg)
    if pipeline_module.resolve_pipeline_plan_only(cfg):
        raise AssertionError(f"actual UI clone should clear seed-only plan flags: {OmegaConf.to_container(cfg, resolve=False)}")
    if cfg.run.grid_run_id != "":
        raise AssertionError(f"actual UI clone should drop the inherited seed grid id: {OmegaConf.to_container(cfg, resolve=False)}")

    seed_cfg = OmegaConf.create(
        {
            "data": {"raw_dataset_id": "REPLACE_WITH_EXISTING_RAW_DATASET_ID"},
            "pipeline": {"plan_only": True},
            "run": {"grid_run_id": "seed__train_ensemble_full"},
        }
    )
    pipeline_module._normalize_ui_cloned_pipeline_cfg(seed_cfg)
    if seed_cfg.pipeline.plan_only is not True:
        raise AssertionError("seed placeholder runs must keep plan_only intact")
    if seed_cfg.run.grid_run_id != "seed__train_ensemble_full":
        raise AssertionError("seed placeholder runs must keep the seed grid id intact")


def _assert_pipeline_run_task_identity_moves_seed_clone_to_run_project() -> None:
    cfg = OmegaConf.create(
        {
            "run": {
                "usecase_id": "ui_new_run_demo",
                "schema_version": "v1",
                "clearml": {
                    "project_root": "LOCAL",
                    "template_set_id": "default",
                },
            },
            "exec_policy": {
                "queues": {
                    "default": "default",
                    "train_model_heavy": "heavy-model",
                }
            },
        }
    )
    captured: dict[str, object] = {}
    originals = {
        "set_project": pipeline_module.set_clearml_task_project,
        "replace_tags": pipeline_module.replace_clearml_task_tags,
        "ensure_tags": pipeline_module.ensure_clearml_task_tags,
        "ensure_properties": pipeline_module.ensure_clearml_task_properties,
    }
    try:
        pipeline_module.set_clearml_task_project = lambda task_id, project_name: captured.setdefault("project", (task_id, project_name)) or True
        pipeline_module.replace_clearml_task_tags = lambda task_id, tags: captured.setdefault("tags", (task_id, list(tags))) or True
        pipeline_module.ensure_clearml_task_tags = lambda task_id, tags: captured.setdefault("system_tags", (task_id, list(tags))) or True
        pipeline_module.ensure_clearml_task_properties = (
            lambda task_id, properties: captured.setdefault("properties", (task_id, dict(properties))) or True
        )
        pipeline_module._apply_pipeline_run_task_identity(
            task_id="seed-clone-task",
            cfg=cfg,
            pipeline_profile="train_ensemble_full",
            metadata={
                "project_name": "LOCAL/TabularAnalysis/Pipelines/Runs/ui_new_run_demo",
                "usecase_id": "ui_new_run_demo",
                "tags": ["task_kind:seed", "usecase:TabularAnalysis", "template:true", "custom:keep"],
                "user_properties": {"grid_run_id": "grid-ui", "base_template_task_id": "seed-base"},
            },
        )
    finally:
        pipeline_module.set_clearml_task_project = originals["set_project"]
        pipeline_module.replace_clearml_task_tags = originals["replace_tags"]
        pipeline_module.ensure_clearml_task_tags = originals["ensure_tags"]
        pipeline_module.ensure_clearml_task_properties = originals["ensure_properties"]
    if captured.get("project") != ("seed-clone-task", "LOCAL/TabularAnalysis/Pipelines/Runs/ui_new_run_demo"):
        raise AssertionError(f"seed clone must move into the pipeline runs project: {captured}")
    tag_payload = captured.get("tags")
    if not isinstance(tag_payload, tuple):
        raise AssertionError(f"runtime tags were not applied: {captured}")
    tags = list(tag_payload[1])
    for required in ("task_kind:run", "usecase:ui_new_run_demo", "pipeline_profile:train_ensemble_full", "custom:keep"):
        if required not in tags:
            raise AssertionError(f"runtime tag normalization missing {required}: {tags}")
    for blocked in ("task_kind:seed", "usecase:TabularAnalysis", "template:true"):
        if blocked in tags:
            raise AssertionError(f"seed clone runtime tags must drop {blocked}: {tags}")
    props_payload = captured.get("properties")
    if not isinstance(props_payload, tuple):
        raise AssertionError(f"runtime properties were not applied: {captured}")
    properties = dict(props_payload[1])
    if properties.get("task_kind") != "run":
        raise AssertionError(f"runtime properties must mark cloned controller as run: {properties}")
    if properties.get("usecase_id") != "ui_new_run_demo":
        raise AssertionError(f"runtime properties must keep the actual UI usecase: {properties}")


def _assert_pipeline_run_summary_tracks_actual_job_statuses() -> None:
    originals = {"get_clearml_task_status": pipeline_module.get_clearml_task_status}
    try:
        statuses = {
            "train-1": "completed",
            "ensemble-1": "failed",
        }
        pipeline_module.get_clearml_task_status = lambda task_id: statuses.get(str(task_id))
        cfg = OmegaConf.create({})
        plan = {
            "plan_only": False,
            "plan_info": {"planned_jobs": 2, "skipped_due_to_policy": 0},
            "preprocess_variants": ["stdscaler_ohe"],
            "model_variants": ["ridge"],
            "max_jobs": 2,
            "max_hpo_trials": 0,
            "hpo_enabled": False,
            "hpo_params_cfg": {},
            "limits": {"max_jobs": 2, "max_models": 2, "max_hpo_trials": 0},
        }
        summary = pipeline_module._build_local_pipeline_run_summary(
            cfg=cfg,
            plan=plan,
            grid_run_id="grid-001",
            dataset_register_ref=None,
            preprocess_refs=[],
            train_refs=[{"task_id": "train-1", "model_variant": "ridge"}],
            train_ensemble_refs=[{"task_id": "ensemble-1", "ensemble_method": "weighted"}],
            leaderboard_ref=None,
            infer_ref=None,
            executed_jobs=2,
        )
        finalized = pipeline_module._finalize_pipeline_run_summary(cfg, summary)
        if finalized.get("status") != "failed":
            raise AssertionError(f"job status aggregation must mark failed pipelines: {finalized}")
        if finalized.get("executed_jobs") != 2:
            raise AssertionError(f"executed_jobs must follow launched child tasks: {finalized}")
        if finalized.get("completed_jobs") != 1 or finalized.get("failed_jobs") != 1:
            raise AssertionError(f"job status counters must reflect child task states: {finalized}")
    finally:
        pipeline_module.get_clearml_task_status = originals["get_clearml_task_status"]


def _assert_manage_templates_pipeline_properties_follow_resolved_context() -> None:
    repo = Path(__file__).resolve().parents[2]
    defaults = manage_templates_module._load_run_defaults(repo)
    ctx = manage_templates_module.PlanContext(
        project_root="LOCAL",
        usecase_id="TabularAnalysis",
        schema_version="v1",
        template_set_id="default",
        solution_root=defaults.solution_root,
        pipeline_seed_namespace=defaults.pipeline_seed_namespace,
        pipeline_root_group=defaults.pipeline_root_group,
        pipeline_templates_group=defaults.pipeline_templates_group,
        pipeline_runs_group=defaults.pipeline_runs_group,
        templates_root_group=defaults.templates_root_group,
        step_templates_group=defaults.step_templates_group,
        runs_root_group=defaults.runs_root_group,
        group_map=dict(defaults.group_map),
    )
    specs = manage_templates_module._load_templates(repo / "conf" / "clearml" / "templates.yaml", ctx)
    spec = next(item for item in specs if item.name == "pipeline")
    resolved = manage_templates_module._resolve_template_spec(
        spec,
        ctx=ctx,
        repo_root=repo,
        repo=None,
        branch=None,
        version_mode="branch",
    )
    if resolved.expected_properties.get("project_root") != "LOCAL":
        raise AssertionError(
            f"pipeline seed properties must follow resolved context overrides: {resolved.expected_properties}"
        )


def _assert_lock_context_payload_keeps_usecase_id() -> None:
    ctx = manage_templates_module.PlanContext(
        project_root="LOCAL",
        usecase_id="TabularAnalysis",
        schema_version="v1",
        template_set_id="default",
        solution_root="TabularAnalysis",
        pipeline_seed_namespace=".pipelines",
        pipeline_root_group="Pipelines",
        pipeline_templates_group="Templates",
        pipeline_runs_group="Runs",
        templates_root_group="Templates",
        step_templates_group="Steps",
        runs_root_group="Runs",
        group_map={},
    )
    payload = manage_templates_module._lock_context_payload(ctx)
    if payload.get("usecase_id") != "TabularAnalysis":
        raise AssertionError(f"lock context must preserve usecase_id: {payload}")
    if payload.get("pipeline_seed_namespace") != ".pipelines":
        raise AssertionError(f"lock context must preserve pipeline_seed_namespace: {payload}")
    if payload.get("group_map") != {}:
        raise AssertionError(f"lock context must preserve group_map: {payload}")


def _assert_live_plan_context_prefers_lock_project_root() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "templates.lock.yaml"
        lock_path.write_text(
            "version: 2\ncontext:\n  project_root: LOCAL\n  usecase_id: TabularAnalysis\n  schema_version: v1\n  template_set_id: default\n  pipeline_seed_namespace: .pipelines\ntemplates: {}\n",
            encoding="utf-8",
        )
        defaults = manage_templates_module.PlanContext(
            project_root="MFG",
            usecase_id="DefaultUsecase",
            schema_version="v9",
            template_set_id="fallback",
            solution_root="TabularAnalysis",
            pipeline_seed_namespace=".pipelines",
            pipeline_root_group="Pipelines",
            pipeline_templates_group="Templates",
            pipeline_runs_group="Runs",
            templates_root_group="Templates",
            step_templates_group="Steps",
            runs_root_group="Runs",
            group_map={},
        )
        ctx = manage_templates_module._resolve_live_plan_context(
            defaults=defaults,
            lock_path=lock_path,
            project_root=None,
            usecase_id=None,
            schema_version=None,
            template_set_id=None,
        )
        if ctx.project_root != "LOCAL" or ctx.usecase_id != "TabularAnalysis":
            raise AssertionError(f"live plan context must follow lock context: {ctx}")
        if ctx.pipeline_seed_namespace != ".pipelines":
            raise AssertionError(f"live plan context must preserve pipeline_seed_namespace: {ctx}")


def _assert_live_plan_context_fails_on_layout_drift() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "templates.lock.yaml"
        lock_path.write_text(
            (
                "version: 2\n"
                "context:\n"
                "  project_root: LOCAL\n"
                "  usecase_id: TabularAnalysis\n"
                "  schema_version: v1\n"
                "  template_set_id: default\n"
                "  solution_root: LegacySolution\n"
                "  pipeline_seed_namespace: legacy.pipelines\n"
                "  pipeline_root_group: Pipelines\n"
                "  pipeline_templates_group: Templates\n"
                "  pipeline_runs_group: Runs\n"
                "  templates_root_group: Templates\n"
                "  step_templates_group: Steps\n"
                "  runs_root_group: Runs\n"
                "  group_map: {}\n"
                "templates: {}\n"
            ),
            encoding="utf-8",
        )
        defaults = manage_templates_module.PlanContext(
            project_root="LOCAL",
            usecase_id="TabularAnalysis",
            schema_version="v1",
            template_set_id="default",
            solution_root="TabularAnalysis",
            pipeline_seed_namespace=".pipelines",
            pipeline_root_group="Pipelines",
            pipeline_templates_group="Templates",
            pipeline_runs_group="Runs",
            templates_root_group="Templates",
            step_templates_group="Steps",
            runs_root_group="Runs",
            group_map={},
        )
        try:
            manage_templates_module._resolve_live_plan_context(
                defaults=defaults,
                lock_path=lock_path,
                project_root=None,
                usecase_id=None,
                schema_version=None,
                template_set_id=None,
            )
        except ValueError as exc:
            if "Lock layout drift detected" not in str(exc):
                raise AssertionError(f"unexpected layout drift error: {exc}")
        else:
            raise AssertionError("live plan context must fail when lock layout drifts")


def _assert_visible_template_graph_mismatch_uses_default_profile_with_explicit_template_id() -> None:
    cfg = OmegaConf.create(
        {
            "run": {
                "clearml": {
                    "pipeline": {
                        "template_task_id": "template-123",
                    }
                }
            }
        }
    )
    profile = resolve_pipeline_profile(
        cfg,
        {
            "run_dataset_register": False,
            "run_preprocess": True,
            "run_train": True,
            "run_train_ensemble": False,
            "run_leaderboard": True,
            "run_infer": False,
            "model_set": None,
        },
    )
    if profile != "pipeline":
        raise AssertionError(f"explicit template_task_id should allow default pipeline profile fallback: {profile}")


def _assert_visible_template_clone_rejects_graph_shaping_model_override() -> None:
    cfg = OmegaConf.create(
        {
            "run": {
                "clearml": {
                    "pipeline": {
                        "template_task_id": "template-123",
                    }
                }
            },
            "pipeline": {
                "run_dataset_register": False,
                "run_preprocess": True,
                "run_train": True,
                "run_train_ensemble": False,
                "run_leaderboard": True,
                "run_infer": False,
                "model_set": "regression_all",
                "model_variants": ["ridge", "lgbm", "xgboost"],
                "grid": {
                    "preprocess_variants": ["stdscaler_ohe"],
                    "model_variants": [],
                },
            },
            "data": {"raw_dataset_id": "ds-001"},
            "ensemble": {"enabled": False},
            "exec_policy": {"limits": {"max_jobs": 50, "max_models": 50, "max_hpo_trials": 0}},
        }
    )
    plan = pipeline_module._build_pipeline_plan(cfg, "grid-graph-check", child_execution="logging")
    try:
        pipeline_module._assert_visible_pipeline_graph_contract(
            cfg=cfg,
            plan=plan,
            pipeline_profile="pipeline",
        )
    except ValueError as exc:
        if "fixed DAG" not in str(exc):
            raise AssertionError(f"unexpected graph mismatch message: {exc}") from exc
        return
    raise AssertionError("visible template clones must reject graph-shaping model overrides")


def _assert_visible_template_clone_allows_selection_subset() -> None:
    cfg = OmegaConf.create(
        {
            "run": {"output_dir": "outputs"},
            "pipeline": {
                "run_dataset_register": False,
                "run_preprocess": True,
                "run_train": True,
                "run_train_ensemble": False,
                "run_leaderboard": True,
                "run_infer": False,
                "model_set": "regression_all",
                "grid": {
                    "preprocess_variants": ["stdscaler_ohe"],
                    "model_variants": [],
                },
                "selection": {
                    "enabled_model_variants": ["ridge", "lgbm", "xgboost"],
                    "enabled_preprocess_variants": ["stdscaler_ohe"],
                },
            },
            "data": {"raw_dataset_id": "ds-001"},
            "ensemble": {"enabled": False},
            "exec_policy": {"limits": {"max_jobs": 50, "max_models": 50, "max_hpo_trials": 0}},
        }
    )
    plan = pipeline_module._build_pipeline_plan(cfg, "grid-selection-check", child_execution="logging")
    pipeline_module._assert_visible_pipeline_graph_contract(
        cfg=cfg,
        plan=plan,
        pipeline_profile="pipeline",
    )
    if plan.get("model_variants") != ["lgbm", "ridge", "xgboost"]:
        raise AssertionError(f"selection subset must shape active model variants only: {plan}")
    requested = (plan.get("selection") or {}).get("requested_model_variants") or []
    expected = pipeline_module._resolve_model_set_variants("regression_all")
    if requested != expected:
        raise AssertionError(f"selection subset must preserve requested model superset: {plan}")
    if int((plan.get("plan_info") or {}).get("disabled_jobs", 0)) <= 0:
        raise AssertionError(f"selection subset must record disabled jobs: {plan}")


def _assert_pipeline_seed_controller_uses_profile_model_set() -> None:
    repo = Path(__file__).resolve().parents[2]
    defaults = manage_templates_module._load_run_defaults(repo)
    ctx = manage_templates_module.PlanContext(
        project_root="LOCAL",
        usecase_id="TabularAnalysis",
        schema_version="v1",
        template_set_id="default",
        solution_root=defaults.solution_root,
        pipeline_seed_namespace=defaults.pipeline_seed_namespace,
        pipeline_root_group=defaults.pipeline_root_group,
        pipeline_templates_group=defaults.pipeline_templates_group,
        pipeline_runs_group=defaults.pipeline_runs_group,
        templates_root_group=defaults.templates_root_group,
        step_templates_group=defaults.step_templates_group,
        runs_root_group=defaults.runs_root_group,
        group_map=dict(defaults.group_map),
    )
    specs = manage_templates_module._load_templates(repo / "conf" / "clearml" / "templates.yaml", ctx)
    spec = next(item for item in specs if item.name == "pipeline")
    cfg = manage_templates_module._compose_pipeline_template_cfg(
        repo,
        spec,
        ctx,
        entry_args=["task=pipeline"],
    )

    class _FakeController:
        def __init__(self) -> None:
            self.parameters: dict[str, object] = {}
            self.steps: list[dict[str, object]] = []
            self._nodes: dict[str, object] = {}

        def add_parameter(self, name: str, default: object = None) -> None:
            self.parameters[str(name)] = default

        def add_step(
            self,
            *,
            name: str,
            parents: object = None,
            parameter_override: object = None,
            execution_queue: object = None,
            clone_base_task: object = None,
            cache_executed_step: object = None,
            recursively_parse_parameters: object = None,
            base_task_id: object = None,
            base_task_factory: object = None,
        ) -> None:
            kwargs = {
                "name": name,
                "parents": parents,
                "parameter_override": parameter_override,
                "execution_queue": execution_queue,
                "clone_base_task": clone_base_task,
                "cache_executed_step": cache_executed_step,
                "recursively_parse_parameters": recursively_parse_parameters,
                "base_task_id": base_task_id,
                "base_task_factory": base_task_factory,
            }
            self.steps.append(dict(kwargs))
            self._nodes[str(name)] = object()

        def create_draft(self) -> None:
            return None

        def _verify(self) -> None:
            return None

        def _serialize_pipeline_task(self) -> None:
            return None

    controller = _FakeController()
    original_resolve_base_task_id = pipeline_module._resolve_base_task_id
    try:
        pipeline_module._resolve_base_task_id = lambda *_args, **_kwargs: "template-base-task"
        draft = pipeline_module.build_pipeline_seed_controller(
            cfg=cfg,
            controller=controller,
            pipeline_profile="pipeline",
        )
    finally:
        pipeline_module._resolve_base_task_id = original_resolve_base_task_id
    if draft["shared_defaults"].get("pipeline.model_set") != "regression_all":
        raise AssertionError(f"pipeline seed defaults must pin regression_all: {draft['shared_defaults']}")
    if draft["shared_defaults"].get("data.raw_dataset_id") != PIPELINE_RAW_DATASET_ID_SENTINEL:
        raise AssertionError(f"pipeline seed defaults must retain the raw dataset placeholder during seed materialization: {draft['shared_defaults']}")
    expected_variants = pipeline_module._resolve_model_set_variants("regression_all")
    if draft["shared_defaults"].get("pipeline.grid.model_variants") != expected_variants:
        raise AssertionError(f"pipeline seed defaults must expand the full regression profile: {draft['shared_defaults']}")
    editable_defaults = draft.get("editable_defaults") or {}
    operator_inputs = draft.get("operator_inputs") or {}
    for key in (
        "run.usecase_id",
        "data.raw_dataset_id",
        "pipeline.selection.enabled_preprocess_variants",
        "pipeline.selection.enabled_model_variants",
    ):
        if key not in editable_defaults:
            raise AssertionError(f"pipeline seed controller must expose {key} as editable default: {draft}")
    step_names = {str(item["name"]) for item in controller.steps}
    expected_train_steps = {f"train__stdscaler_ohe__{variant}" for variant in expected_variants}
    missing = sorted(expected_train_steps - step_names)
    if missing:
        raise AssertionError(f"pipeline seed controller must materialize the full regression graph: missing {missing}")
    for required in ("preprocess__stdscaler_ohe", "leaderboard"):
        if required not in step_names:
            raise AssertionError(f"pipeline seed controller missing {required}: {sorted(step_names)}")
    if "pipeline_run" not in draft or "report_bundle" not in draft:
        raise AssertionError(f"pipeline seed controller must include pipeline report materialization payloads: {draft.keys()}")
    if (((operator_inputs.get("pipeline") or {}).get("selection") or {}).get("enabled_model_variants")) != expected_variants:
        raise AssertionError(f"operator inputs must mirror editable model selection: {operator_inputs}")


def _assert_pipeline_operator_inputs_reject_placeholder_for_actual_runs() -> None:
    placeholder_cfg = OmegaConf.create(
        {
            "task": {"name": "pipeline"},
            "run": {"usecase_id": "demo_usecase"},
            "data": {"raw_dataset_id": PIPELINE_RAW_DATASET_ID_SENTINEL},
            "pipeline": {
                "profile": "pipeline",
                "run_dataset_register": False,
                "run_preprocess": True,
                "run_train": True,
                "run_train_ensemble": False,
                "run_leaderboard": True,
                "run_infer": False,
            },
        }
    )
    try:
        validate_pipeline_operator_inputs(placeholder_cfg, allow_placeholder_raw_dataset=False)
    except ValueError as exc:
        if "data.raw_dataset_id is still the seed placeholder" not in str(exc):
            raise AssertionError(f"unexpected placeholder validation error: {exc}")
    else:
        raise AssertionError("actual pipeline runs must reject the seed raw_dataset placeholder")
    validate_pipeline_operator_inputs(placeholder_cfg, allow_placeholder_raw_dataset=True)

    empty_cfg = OmegaConf.create(
        {
            "task": {"name": "pipeline"},
            "run": {"usecase_id": "demo_usecase"},
            "data": {"raw_dataset_id": ""},
            "pipeline": {
                "profile": "pipeline",
                "run_dataset_register": False,
                "run_preprocess": True,
                "run_train": True,
                "run_train_ensemble": False,
                "run_leaderboard": True,
                "run_infer": False,
            },
        }
    )
    try:
        validate_pipeline_operator_inputs(empty_cfg, allow_placeholder_raw_dataset=False)
    except ValueError as exc:
        if "data.raw_dataset_id is empty" not in str(exc):
            raise AssertionError(f"unexpected empty raw_dataset_id validation error: {exc}")
    else:
        raise AssertionError("actual pipeline runs must reject an empty raw_dataset_id when dataset_register is disabled")


def _assert_rehearsal_sync_pipeline_artifacts_missing_only_reuses_existing_outputs() -> None:
    import tabular_analysis.platform_adapter_task as platform_task_module

    original_get_copy = platform_task_module.get_task_artifact_local_copy
    original_list_names = rehearsal_module._list_task_artifact_names
    fetch_calls: list[str] = []
    try:
        platform_task_module.get_task_artifact_local_copy = lambda cfg, task_id, name: fetch_calls.append(str(name))  # type: ignore[assignment]
        rehearsal_module._list_task_artifact_names = lambda task_id: {
            "pipeline_run.json",
            "run_summary.json",
            "report.json",
            "report.md",
            "report_links.json",
            "out.json",
            "manifest.json",
            "config_resolved.yaml",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            stage_dir = output_dir / "99_pipeline"
            stage_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "pipeline_run.json",
                "run_summary.json",
                "report.json",
                "report.md",
                "report_links.json",
                "out.json",
                "manifest.json",
                "config_resolved.yaml",
            ):
                (stage_dir / name).write_text("{}", encoding="utf-8")
            copied = rehearsal_module._sync_pipeline_artifacts(
                _REPO,
                output_dir,
                "pipeline-task",
                missing_only=True,
            )
            if set(copied) != {
                "pipeline_run.json",
                "run_summary.json",
                "report.json",
                "report.md",
                "report_links.json",
                "out.json",
                "manifest.json",
                "config_resolved.yaml",
            }:
                raise AssertionError(f"missing_only sync should reuse existing outputs: {copied}")
            if fetch_calls:
                raise AssertionError(f"missing_only sync should not fetch files that already exist: {fetch_calls}")
    finally:
        platform_task_module.get_task_artifact_local_copy = original_get_copy
        rehearsal_module._list_task_artifact_names = original_list_names


def _assert_lock_entry_preserves_updated_at_without_identity_change() -> None:
    existing = {
        "task_id": "task-123",
        "project_name": "LOCAL/TabularAnalysis/.pipelines/pipeline",
        "task_name": "pipeline",
        "entrypoint": "python tools/clearml_entrypoint.py task=pipeline",
        "kind": "seed",
        "updated_at": "2026-04-04T17:33:36Z",
    }
    payload = manage_templates_module._build_lock_template_entry(
        existing,
        task_id="task-123",
        project_name="LOCAL/TabularAnalysis/.pipelines/pipeline",
        task_name="pipeline",
        entrypoint="python tools/clearml_entrypoint.py task=pipeline",
        now="2026-04-05T00:00:00Z",
        kind="seed",
    )
    if payload.get("updated_at") != "2026-04-04T17:33:36Z":
        raise AssertionError(f"lock entry should preserve updated_at when identity is unchanged: {payload}")
    changed = manage_templates_module._build_lock_template_entry(
        existing,
        task_id="task-999",
        project_name="LOCAL/TabularAnalysis/.pipelines/pipeline",
        task_name="pipeline",
        entrypoint="python tools/clearml_entrypoint.py task=pipeline",
        now="2026-04-05T00:00:00Z",
        kind="seed",
    )
    if changed.get("updated_at") != "2026-04-05T00:00:00Z":
        raise AssertionError(f"lock entry should refresh updated_at when identity changes: {changed}")


def _assert_pipeline_report_primary_summary_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        dataset_dir = root / "01_dataset_register"
        preprocess_dir = root / "02_preprocess"
        leaderboard_dir = root / "99_leaderboard"
        for path in (dataset_dir, preprocess_dir, leaderboard_dir):
            path.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "out.json").write_text(
            json.dumps(
                {
                    "raw_dataset_id": "raw-123",
                    "raw_schema": {
                        "rows": 12,
                        "columns": ["num1", "num2"],
                        "target_column": "target",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (preprocess_dir / "out.json").write_text(
            json.dumps(
                {
                    "processed_dataset_id": "processed-123",
                    "split_hash": "split-123",
                    "recipe_hash": "recipe-123",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (preprocess_dir / "schema.json").write_text(
            json.dumps(
                {
                    "rows": 12,
                    "columns": ["num1", "num2"],
                    "target_column": "target",
                    "id_columns": [],
                    "drop_columns": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (preprocess_dir / "split.json").write_text(
            json.dumps(
                {"strategy": "random", "test_size": 0.2, "seed": 42},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (preprocess_dir / "recipe.json").write_text(
            json.dumps({"variant": {"name": "stdscaler_ohe"}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (leaderboard_dir / "leaderboard.csv").write_text(
            (
                "rank,model_variant,preprocess_variant,primary_metric,best_score,model_id,train_task_ref\n"
                "1,ridge,stdscaler_ohe,rmse,0.10,model-ridge,train-1\n"
                "2,lgbm,stdscaler_ohe,rmse,0.20,model-lgbm,train-2\n"
                "3,xgboost,stdscaler_ohe,rmse,0.30,model-xgb,train-3\n"
            ),
            encoding="utf-8",
        )
        (leaderboard_dir / "out.json").write_text(
            json.dumps(
                {
                    "recommended_primary_metric": "rmse",
                    "recommended_best_score": 0.10,
                    "recommended_train_task_ref": "train-1",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (leaderboard_dir / "recommendation.json").write_text(
            json.dumps(
                {
                    "recommended_primary_metric": "rmse",
                    "recommended_best_score": 0.10,
                    "recommended_train_task_ref": "train-1",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (leaderboard_dir / "decision_summary.json").write_text(
            json.dumps(
                {
                    "comparability": {
                        "processed_dataset_id": "processed-123",
                        "split_hash": "split-123",
                        "recipe_hash": "recipe-123",
                        "primary_metric": "rmse",
                        "direction": "minimize",
                        "task_type": "regression",
                    },
                    "recommended": {
                        "primary_metric": "rmse",
                        "best_score": 0.10,
                        "task_type": "regression",
                        "train_task_ref": "train-1",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        pipeline_run = {
            "grid_run_id": "grid123",
            "status": "completed",
            "requested_jobs": 4,
            "planned_jobs": 2,
            "executed_jobs": 2,
            "disabled_jobs": 2,
            "completed_jobs": 2,
            "failed_jobs": 0,
            "stopped_jobs": 0,
            "running_jobs": 0,
            "queued_jobs": 0,
            "skipped_due_to_policy": 0,
            "plan_only": False,
            "dataset_register_ref": {"run_dir": str(dataset_dir)},
            "preprocess_ref": [
                {
                    "run_dir": str(preprocess_dir),
                    "preprocess_variant": "stdscaler_ohe",
                    "processed_dataset_id": "processed-123",
                    "split_hash": "split-123",
                    "recipe_hash": "recipe-123",
                }
            ],
            "train_refs": [
                {
                    "task_id": "train-1",
                    "train_task_id": "train-1",
                    "model_variant": "ridge",
                    "preprocess_variant": "stdscaler_ohe",
                    "best_score": 0.10,
                    "primary_metric": "rmse",
                },
                {
                    "task_id": "train-2",
                    "train_task_id": "train-2",
                    "model_variant": "lgbm",
                    "preprocess_variant": "stdscaler_ohe",
                    "best_score": 0.20,
                    "primary_metric": "rmse",
                },
            ],
            "train_ensemble_refs": [],
            "leaderboard_ref": {"run_dir": str(leaderboard_dir)},
            "infer_ref": None,
            "grid": {
                "preprocess_variants": ["stdscaler_ohe"],
                "model_variants": ["ridge", "lgbm"],
                "hpo": {"enabled": False, "params": {}},
            },
            "selection": {
                "requested_preprocess_variants": ["stdscaler_ohe"],
                "active_preprocess_variants": ["stdscaler_ohe"],
                "disabled_preprocess_variants": [],
                "requested_model_variants": ["ridge", "lgbm"],
                "active_model_variants": ["ridge", "lgbm"],
                "disabled_model_variants": [],
                "requested_ensemble_methods": [],
                "active_ensemble_methods": [],
                "disabled_ensemble_methods": [],
            },
            "disabled_selection": [],
        }
        bundle = pipeline_report_module.build_pipeline_report_bundle(
            pipeline_run,
            cfg=None,
            max_models=5,
            pipeline_run_dir=root,
            pipeline_task_id="pipeline-task",
        )
        summary = bundle.payload.get("summary") or {}
        if summary.get("executed_jobs") != 2:
            raise AssertionError(f"pipeline report must use executed_jobs as the primary launched-work field: {summary}")
        if summary.get("ranked_candidates") != 3:
            raise AssertionError(f"pipeline report must derive ranked_candidates from leaderboard rows: {summary}")
        if summary.get("models_tried") != 2:
            raise AssertionError(f"pipeline report must keep models_tried as the legacy executed_jobs mirror: {summary}")
        for required in ("- executed_jobs: 2", "- ranked_candidates: 3", "- models_tried_legacy: 2"):
            if required not in bundle.markdown:
                raise AssertionError(f"pipeline report markdown missing {required!r}: {bundle.markdown}")


def _assert_uv_lock_exposes_split_model_extras() -> None:
    text = (_REPO / "uv.lock").read_text(encoding="utf-8")
    for required in (
        'lightgbm = [',
        'xgboost = [',
        'catboost = [',
        'provides-extras = ["api", "models", "lightgbm", "xgboost", "catboost", "tabpfn", "optuna", "dev"]',
        "extra == 'lightgbm'",
        "extra == 'xgboost'",
        "extra == 'catboost'",
    ):
        if required not in text:
            raise AssertionError(f"uv.lock missing split extra contract: {required}")


def main() -> int:
    _assert_clearml_hocon_reader()
    _assert_batch_execution_mode()
    _assert_rehearsal_sync_cfg_keeps_clearml_enabled()
    _assert_strict_template_lookup()
    _assert_visible_pipeline_seed_lookup()
    _assert_project_layout_contract()
    _assert_runtime_tag_filter_contract()
    _assert_task_time_extras()
    _assert_entrypoint_reads_clearml_slash_overrides()
    _assert_reset_clearml_task_args_replaces_stale_args()
    _assert_regression_model_set_contract()
    _assert_explicit_pipeline_variants_override_model_set()
    _assert_pipeline_profile_defaults_clear_stale_model_variants()
    _assert_pipeline_profile_defaults_clear_stale_selection()
    _assert_pipeline_profile_sets_run_flags()
    _assert_pipeline_controller_context_attaches_by_task_id()
    _assert_pipeline_controller_context_attaches_by_pipeline_task_id_override()
    _assert_rehearsal_rebuild_pipeline_outputs_from_clearml()
    _assert_pipeline_step_references_are_not_quoted()
    _assert_pipeline_steps_enable_recursive_parameter_parsing()
    _assert_leaderboard_bootstrap_extras_follow_pipeline_model_variants()
    _assert_loaded_pipeline_controller_reseeds_runtime_defaults()
    _assert_plan_only_controller_does_not_launch_steps()
    _assert_pipeline_template_defaults_keep_plan_only()
    _assert_ui_clone_cfg_normalization_clears_seed_only_defaults()
    _assert_pipeline_run_task_identity_moves_seed_clone_to_run_project()
    _assert_pipeline_run_summary_tracks_actual_job_statuses()
    _assert_manage_templates_pipeline_properties_follow_resolved_context()
    _assert_lock_context_payload_keeps_usecase_id()
    _assert_live_plan_context_prefers_lock_project_root()
    _assert_live_plan_context_fails_on_layout_drift()
    _assert_visible_template_graph_mismatch_uses_default_profile_with_explicit_template_id()
    _assert_visible_template_clone_rejects_graph_shaping_model_override()
    _assert_visible_template_clone_allows_selection_subset()
    _assert_pipeline_seed_controller_uses_profile_model_set()
    _assert_pipeline_operator_inputs_reject_placeholder_for_actual_runs()
    _assert_rehearsal_sync_pipeline_artifacts_missing_only_reuses_existing_outputs()
    _assert_lock_entry_preserves_updated_at_without_identity_change()
    _assert_pipeline_report_primary_summary_fields()
    _assert_uv_lock_exposes_split_model_extras()
    print("OK: clearml runtime contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
