"""
Background thread orchestration for interactive active learning.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import torch
from torch.utils.data import DataLoader

from active_loop import ActiveLearningLoop
from data_manager import ALDataManager
from dataloader import get_datasets
from events import Event, EventType
from experiment_state import ExperimentState
from models import get_model
from strategies import get_strategy
from trainer import Trainer

logger = logging.getLogger(__name__)


def build_al_loop(config: Any, run_dir: Path) -> ActiveLearningLoop:
    """Build the AL loop once per run."""
    exp_dir = Path(run_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    datasets = get_datasets(
        data_dir=config.data.data_dir,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        augmentation=config.data.augmentation,
        seed=config.experiment.seed,
    )

    train_dataset = datasets["train_dataset"]
    class_names = datasets["class_names"]
    pin = torch.cuda.is_available()

    val_loader = DataLoader(
        datasets["val_dataset"],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        datasets["test_dataset"],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin,
    )

    if config.model.num_classes is None:
        config.model.num_classes = len(class_names)

    model = get_model(
        name=config.model.name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        device=config.experiment.device,
    )

    data_manager = ALDataManager(
        dataset=train_dataset,
        initial_pool_size=config.active_learning.initial_pool_size,
        seed=config.experiment.seed,
        exp_dir=exp_dir,
    )

    trainer = Trainer(
        model=model,
        config=config,
        exp_dir=exp_dir,
        device=config.experiment.device,
    )

    strategy = get_strategy(
        config.active_learning.sampling_strategy,
        config.active_learning.uncertainty_method,
    )

    return ActiveLearningLoop(
        trainer=trainer,
        data_manager=data_manager,
        strategy=strategy,
        val_loader=val_loader,
        test_loader=test_loader,
        exp_dir=exp_dir,
        config=config,
        class_names=class_names,
    )


def _emit_event(
    state: ExperimentState,
    event_type: EventType,
    run_id: str,
    cycle: int = 0,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    state.inbox.put(
        Event(
            type=event_type,
            run_id=run_id,
            cycle=cycle,
            data=data or {},
        )
    )


def _should_abort(state: ExperimentState, run_id: str) -> bool:
    if state.stop_event.is_set():
        return True
    return not state.is_run_active(run_id)


def _as_dict(value: Any) -> Dict[str, Any]:
    """Best-effort normalization of metrics objects to plain dict."""
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, dict):
            return dict(converted)
    if hasattr(value, "model_dump"):
        converted = value.model_dump()
        if isinstance(converted, dict):
            return dict(converted)
    return {}


def _serialize_probe_images(al_loop: ActiveLearningLoop) -> List[Dict[str, Any]]:
    """Serialize probe images to plain dictionaries for event transport."""
    probe_images = list(getattr(al_loop, "probe_images", []))
    serialized: List[Dict[str, Any]] = []
    for probe in probe_images:
        if hasattr(probe, "to_dict"):
            data = probe.to_dict()
            if isinstance(data, dict):
                serialized.append(dict(data))
                continue
        if isinstance(probe, dict):
            serialized.append(dict(probe))
    return serialized


def _as_named_distribution(dist: Dict[int, int], class_names: List[str]) -> Dict[str, int]:
    """Convert class-index distribution to readable class-name keys."""
    named: Dict[str, int] = {}
    for class_idx, count in sorted(dist.items(), key=lambda item: int(item[0])):
        idx = int(class_idx)
        key = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
        named[key] = int(count)
    return named


def _pool_stats(al_loop: ActiveLearningLoop) -> Dict[str, Any]:
    """Extract current pool sizes and class distributions."""
    class_names = list(getattr(al_loop, "class_names", []))
    pool_info = al_loop.data_manager.get_pool_info()
    labeled_dist = al_loop.data_manager.get_class_distribution(pool="labeled")
    unlabeled_dist = al_loop.data_manager.get_class_distribution(pool="unlabeled")
    return {
        "labeled_pool_size": int(pool_info.get("labeled", 0)),
        "unlabeled_pool_size": int(pool_info.get("unlabeled", 0)),
        "labeled_class_distribution": _as_named_distribution(labeled_dist, class_names),
        "unlabeled_class_distribution": _as_named_distribution(unlabeled_dist, class_names),
    }


def _wait_for_next_step(state: ExperimentState, run_id: str) -> bool:
    """Block until next-step signal or abort request."""
    while True:
        if _should_abort(state, run_id):
            return False
        if state.next_step_event.wait(timeout=0.5):
            state.next_step_event.clear()
            return True
        state.touch_heartbeat(run_id)


def _flush_artifacts(al_loop: Optional[ActiveLearningLoop], run_dir: Path) -> None:
    """Best-effort save of all run artifacts."""
    if al_loop is None:
        return
    try:
        al_loop._save_results()
        al_loop.data_manager.save_state(Path(run_dir) / "al_pool_state.json")
        al_loop.trainer.save_training_log()
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to flush artifacts")


def _exit_stopped(
    state: ExperimentState,
    run_id: str,
    cycle: int,
    al_loop: Optional[ActiveLearningLoop],
    run_dir: Path,
) -> None:
    _flush_artifacts(al_loop, run_dir)
    if state.is_run_active(run_id):
        _emit_event(state, EventType.RUN_STOPPED, run_id=run_id, cycle=cycle, data={})


def run_experiment(state: ExperimentState, config: Any, run_dir: Path) -> None:
    """
    Entry point for the backend daemon thread.
    """
    local_run_id = state.snapshot()["run_id"]
    al_loop: Optional[ActiveLearningLoop] = None
    current_cycle = 0

    try:
        al_loop = build_al_loop(config, run_dir)
        state.touch_heartbeat(local_run_id)

        if _should_abort(state, local_run_id):
            _exit_stopped(state, local_run_id, current_cycle, al_loop, run_dir)
            return

        total_cycles = int(config.active_learning.num_cycles)
        auto_annotate = bool(config.active_learning.auto_annotate)
        step_mode = bool(getattr(config.active_learning, "step_mode", False))

        for cycle in range(1, total_cycles + 1):
            if step_mode and cycle > 1:
                _emit_event(
                    state,
                    EventType.WAITING_FOR_STEP,
                    run_id=local_run_id,
                    cycle=cycle,
                    data={"next_cycle": cycle, "total_cycles": total_cycles},
                )
                if not _wait_for_next_step(state, local_run_id):
                    _exit_stopped(state, local_run_id, current_cycle, al_loop, run_dir)
                    return

            current_cycle = cycle
            if _should_abort(state, local_run_id):
                _exit_stopped(state, local_run_id, cycle, al_loop, run_dir)
                return

            prep_info = al_loop.prepare_cycle(cycle)
            pool_stats = _pool_stats(al_loop)
            class_names = list(getattr(al_loop, "class_names", []))
            _emit_event(
                state,
                EventType.CYCLE_STARTED,
                run_id=local_run_id,
                cycle=cycle,
                data={
                    "cycle": cycle,
                    "total_cycles": total_cycles,
                    "class_names": class_names,
                    "labeled_pool_size": int(prep_info.get("labeled_count", 0)),
                    "unlabeled_pool_size": int(prep_info.get("unlabeled_count", 0)),
                    "labeled_class_distribution": pool_stats["labeled_class_distribution"],
                    "unlabeled_class_distribution": pool_stats["unlabeled_class_distribution"],
                },
            )

            epochs = int(config.training.epochs)
            for epoch in range(1, epochs + 1):
                if _should_abort(state, local_run_id):
                    _exit_stopped(state, local_run_id, cycle, al_loop, run_dir)
                    return

                state.touch_heartbeat(local_run_id)

                metrics = al_loop.train_single_epoch(epoch)
                metrics_dict = _as_dict(metrics)
                if not metrics_dict:
                    metrics_dict = {"epoch": epoch}
                _emit_event(
                    state,
                    EventType.EPOCH_DONE,
                    run_id=local_run_id,
                    cycle=cycle,
                    data={
                        "epoch": epoch,
                        "total_epochs": epochs,
                        "metrics": metrics_dict,
                    },
                )

                if _should_abort(state, local_run_id):
                    _exit_stopped(state, local_run_id, cycle, al_loop, run_dir)
                    return

                if al_loop.should_stop_early():
                    break

            if _should_abort(state, local_run_id):
                _exit_stopped(state, local_run_id, cycle, al_loop, run_dir)
                return

            state.touch_heartbeat(local_run_id)

            test_metrics = al_loop.run_evaluation()
            cycle_metrics = al_loop.finalize_cycle(test_metrics).model_dump()
            probe_images = _serialize_probe_images(al_loop)
            pool_stats = _pool_stats(al_loop)
            _emit_event(
                state,
                EventType.EVAL_COMPLETE,
                run_id=local_run_id,
                cycle=cycle,
                data={
                    "cycle_metrics": cycle_metrics,
                    "probe_images": probe_images,
                    "labeled_class_distribution": pool_stats["labeled_class_distribution"],
                    "unlabeled_class_distribution": pool_stats["unlabeled_class_distribution"],
                },
            )

            if cycle >= total_cycles:
                continue

            if int(cycle_metrics.get("unlabeled_pool_size", 0)) <= 0:
                break

            if _should_abort(state, local_run_id):
                _exit_stopped(state, local_run_id, cycle, al_loop, run_dir)
                return

            _emit_event(
                state,
                EventType.QUERYING_STARTED,
                run_id=local_run_id,
                cycle=cycle,
                data={"cycle": cycle},
            )
            state.touch_heartbeat(local_run_id)

            if auto_annotate:
                if _should_abort(state, local_run_id):
                    _exit_stopped(state, local_run_id, cycle, al_loop, run_dir)
                    return

                summary = al_loop.query_and_auto_annotate(
                    heartbeat_fn=lambda: state.touch_heartbeat(local_run_id)
                )
                if int(summary.get("queried_count", 0)) <= 0:
                    continue

                pool_stats = _pool_stats(al_loop)
                _emit_event(
                    state,
                    EventType.ANNOTATIONS_APPLIED,
                    run_id=local_run_id,
                    cycle=cycle,
                    data={
                        "count": int(summary.get("applied_count", 0)),
                        "labeled_pool_size": pool_stats["labeled_pool_size"],
                        "unlabeled_pool_size": pool_stats["unlabeled_pool_size"],
                        "labeled_class_distribution": pool_stats["labeled_class_distribution"],
                        "unlabeled_class_distribution": pool_stats["unlabeled_class_distribution"],
                    },
                )
                state.clear_annotations()
                continue

            queried_images = al_loop.query_samples(
                heartbeat_fn=lambda: state.touch_heartbeat(local_run_id)
            )
            queried_dicts = [img.to_dict() for img in queried_images]

            if not queried_dicts:
                continue

            state.clear_annotations()
            _emit_event(
                state,
                EventType.NEW_IMAGES,
                run_id=local_run_id,
                cycle=cycle,
                data={
                    "queried_images": queried_dicts,
                    "query_token": str(uuid4()),
                },
            )

            annotations: List[Dict[str, Any]]
            while True:
                if _should_abort(state, local_run_id):
                    _exit_stopped(state, local_run_id, cycle, al_loop, run_dir)
                    return
                if state.annotations_ready.wait(timeout=0.5):
                    annotations = state.consume_annotations(local_run_id, cycle)
                    if annotations is None:
                        continue
                    break
                state.touch_heartbeat(local_run_id)

            if _should_abort(state, local_run_id):
                _exit_stopped(state, local_run_id, cycle, al_loop, run_dir)
                return

            al_loop.receive_annotations(annotations)
            pool_stats = _pool_stats(al_loop)
            _emit_event(
                state,
                EventType.ANNOTATIONS_APPLIED,
                run_id=local_run_id,
                cycle=cycle,
                data={
                    "count": len(annotations),
                    "labeled_pool_size": pool_stats["labeled_pool_size"],
                    "unlabeled_pool_size": pool_stats["unlabeled_pool_size"],
                    "labeled_class_distribution": pool_stats["labeled_class_distribution"],
                    "unlabeled_class_distribution": pool_stats["unlabeled_class_distribution"],
                },
            )

        _flush_artifacts(al_loop, run_dir)
        if state.is_run_active(local_run_id):
            _emit_event(
                state,
                EventType.RUN_FINISHED,
                run_id=local_run_id,
                cycle=current_cycle,
                data={},
            )

    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Backend thread failed")
        _flush_artifacts(al_loop, run_dir)
        if state.is_run_active(local_run_id):
            _emit_event(
                state,
                EventType.RUN_ERROR,
                run_id=local_run_id,
                cycle=current_cycle,
                data={
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
