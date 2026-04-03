#!/usr/bin/env python3
"""Regression checks for ClearML runtime contracts."""

from __future__ import annotations

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
from tabular_analysis.processes import pipeline as pipeline_module
from tabular_analysis.processes.infer_support import resolve_batch_execution_mode
from tabular_analysis.registry.models import list_model_variants
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
        template_module.clearml_task_tags = lambda task_obj: ["template:true", "usecase:TabularAnalysis", "process:infer", "schema:v1", "template_set:default", "solution:tabular-analysis"]
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


def _assert_visible_pipeline_template_lookup() -> None:
    calls: list[list[str]] = []
    original = {
        "list": pipeline_template_module.list_clearml_tasks_by_tags,
        "status": pipeline_template_module.clearml_task_status_from_obj,
        "tags": pipeline_template_module.clearml_task_tags,
        "script": pipeline_template_module.clearml_task_script,
        "id": pipeline_template_module.clearml_task_id,
        "mismatch": pipeline_template_module.clearml_script_mismatches,
        "spec": pipeline_template_module.resolve_clearml_script_spec,
    }

    class _FakeTask:
        pass

    task = _FakeTask()

    try:
        pipeline_template_module.list_clearml_tasks_by_tags = lambda tags, project_name=None: calls.append(list(tags)) or [task]
        pipeline_template_module.clearml_task_status_from_obj = lambda task_obj: "created"
        pipeline_template_module.clearml_task_tags = lambda task_obj: [
            "template:true",
            "usecase:TabularAnalysis",
            "process:pipeline",
            "schema:v1",
            "template_set:default",
            "solution:tabular-analysis",
            "task_kind:template",
            "pipeline_profile:pipeline",
        ]
        pipeline_template_module.clearml_task_script = lambda task_obj: {"entry_point": "tools/clearml_entrypoint.py"}
        pipeline_template_module.clearml_task_id = lambda task_obj: "pipeline-template-123"
        pipeline_template_module.clearml_script_mismatches = lambda expected, actual: False
        pipeline_template_module.resolve_clearml_script_spec = lambda *args, **kwargs: {"entry_point": "tools/clearml_entrypoint.py"}
        task_id = pipeline_template_module.resolve_pipeline_template_task_id(
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
        if task_id != "pipeline-template-123":
            raise AssertionError(f"unexpected pipeline template id: {task_id}")
        if len(calls) != 1:
            raise AssertionError(f"pipeline template lookup should be strict and single-pass: {calls}")
        tags = calls[0]
        for required in ("process:pipeline", "task_kind:template", "pipeline_profile:pipeline", "template_set:default", "schema:v1"):
            if required not in tags:
                raise AssertionError(f"visible pipeline lookup missing {required}: {tags}")
    finally:
        pipeline_template_module.list_clearml_tasks_by_tags = original["list"]
        pipeline_template_module.clearml_task_status_from_obj = original["status"]
        pipeline_template_module.clearml_task_tags = original["tags"]
        pipeline_template_module.clearml_task_script = original["script"]
        pipeline_template_module.clearml_task_id = original["id"]
        pipeline_template_module.clearml_script_mismatches = original["mismatch"]
        pipeline_template_module.resolve_clearml_script_spec = original["spec"]


def _assert_task_time_extras() -> None:
    if resolve_required_uv_extras(task_name="dataset_register") != []:
        raise AssertionError("dataset_register should not request optional extras")
    if resolve_required_uv_extras(task_name="preprocess") != []:
        raise AssertionError("preprocess should not request optional extras")
    if resolve_required_uv_extras(task_name="train_model", model_variant_name="lgbm") != ["models"]:
        raise AssertionError("lgbm train should use models extra only")
    if resolve_required_uv_extras(task_name="train_model", model_variant_name="ridge") != []:
        raise AssertionError("ridge train should not request optional extras")
    if resolve_required_uv_extras(task_name="infer", model_variant_name="lgbm") != ["models"]:
        raise AssertionError("lgbm infer should use models extra only")
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
    loaded: dict[str, str] = {}
    _store_loaded_override(loaded, "data/raw_dataset_id", "template_raw_dataset")
    _store_loaded_override(loaded, "data.raw_dataset_id", "runtime_dataset")
    _store_loaded_override(loaded, "default_queue", "default")
    _store_loaded_override(loaded, "pipeline/profile", "train_ensemble_full")
    if loaded.get("data.raw_dataset_id") != "runtime_dataset":
        raise AssertionError(f"dotted override must win over slash placeholder: {loaded}")
    if "default_queue" in loaded:
        raise AssertionError(f"controller-only default_queue should not be forwarded to Hydra: {loaded}")
    if loaded.get("+pipeline.profile") != "train_ensemble_full":
        raise AssertionError(f"runtime-only pipeline profile should be appended, not overridden: {loaded}")
    if _extract_cli_keys(["+pipeline.model_set=regression_all", "task=pipeline"]) != {"pipeline.model_set", "task"}:
        raise AssertionError("CLI key extraction must treat +override and plain override as the same key")


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
        "svr",
        "xgboost",
    ]
    if variants != expected:
        raise AssertionError(f"unexpected regression_all variants: {variants}")
    regression_variants = list_model_variants("regression")
    classification_variants = list_model_variants("classification")
    if "svc" in regression_variants:
        raise AssertionError(f"svc must not be selectable for regression: {regression_variants}")
    if "svr" not in regression_variants:
        raise AssertionError(f"svr must remain selectable for regression: {regression_variants}")
    if "svc" not in classification_variants:
        raise AssertionError(f"svc must remain selectable for classification: {classification_variants}")
    if "svr" in classification_variants:
        raise AssertionError(f"svr must not be selectable for classification: {classification_variants}")


def _assert_pipeline_controller_context_attaches_by_task_id() -> None:
    class _FakeTaskObject:
        name = "pipeline"

        def get_project_name(self) -> str:
            return "LOCAL/TabularAnalysis/Test/00_Pipelines"

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
                        "clearml": {"project_name": "LOCAL/TabularAnalysis/Test/00_Pipelines"},
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


def _assert_loaded_pipeline_controller_reseeds_runtime_defaults() -> None:
    class _FakeTask:
        pass

    class _FakeController:
        def __init__(self) -> None:
            self._nodes = {"preprocess": object()}
            self.started_with: str | None = "unset"

        def start(self, *, queue: str | None = None) -> None:
            self.started_with = queue

    fake_controller = _FakeController()
    fake_task = _FakeTask()
    originals = {
        "resolve_contract": pipeline_module._resolve_visible_pipeline_run_contract,
        "clearml_task_id": pipeline_module.clearml_task_id,
        "apply_defaults": pipeline_module._apply_visible_pipeline_run_defaults,
        "load_controller": pipeline_module.load_pipeline_controller_from_task,
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
            queue_name="services",
        )
        pipeline_module.clearml_task_id = lambda task: "controller-123"
        pipeline_module._apply_visible_pipeline_run_defaults = lambda **kwargs: {"run.grid_run_id": "grid-001"}
        pipeline_module.load_pipeline_controller_from_task = lambda source_task: fake_controller
        pipeline_module._collect_step_task_ids = lambda controller: {}
        pipeline_module._build_clearml_pipeline_refs = lambda **kwargs: (None, [], [], [], None, None)
        pipeline_module._build_local_pipeline_run_summary = lambda **kwargs: {"status": "stub"}
        cfg = OmegaConf.create({"run": {"clearml": {"queue_name": "default"}}})
        ctx = pipeline_module.TaskContext(task=fake_task, project_name="LOCAL/TabularAnalysis/Test/00_Pipelines", task_name="pipeline", output_dir=Path.cwd())
        summary = pipeline_module._execute_current_pipeline_controller(cfg=cfg, ctx=ctx, grid_run_id="grid-001")
        if fake_controller.started_with is not None:
            raise AssertionError(f"remote controller should start in-place without re-enqueueing itself: {fake_controller.started_with}")
        if getattr(fake_controller, "_default_execution_queue", None) != "default":
            raise AssertionError(f"loaded controller should reseed default execution queue: {getattr(fake_controller, '_default_execution_queue', None)}")
        pipeline_args = getattr(fake_controller, "_pipeline_args", {})
        if pipeline_args.get("run/grid_run_id") != "grid-001":
            raise AssertionError(f"loaded controller should reseed pipeline args: {pipeline_args}")
        if summary.get("pipeline_task_id") != "controller-123":
            raise AssertionError(f"unexpected pipeline summary: {summary}")
    finally:
        pipeline_module._resolve_visible_pipeline_run_contract = originals["resolve_contract"]
        pipeline_module.clearml_task_id = originals["clearml_task_id"]
        pipeline_module._apply_visible_pipeline_run_defaults = originals["apply_defaults"]
        pipeline_module.load_pipeline_controller_from_task = originals["load_controller"]
        pipeline_module._collect_step_task_ids = originals["collect_step_ids"]
        pipeline_module._build_clearml_pipeline_refs = originals["build_refs"]
        pipeline_module._build_local_pipeline_run_summary = originals["build_summary"]


def main() -> int:
    _assert_clearml_hocon_reader()
    _assert_batch_execution_mode()
    _assert_strict_template_lookup()
    _assert_visible_pipeline_template_lookup()
    _assert_task_time_extras()
    _assert_entrypoint_reads_clearml_slash_overrides()
    _assert_regression_model_set_contract()
    _assert_pipeline_controller_context_attaches_by_task_id()
    _assert_loaded_pipeline_controller_reseeds_runtime_defaults()
    print("OK: clearml runtime contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
