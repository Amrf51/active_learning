"""
Background thread orchestration for interactive active learning.
"""

from __future__ import annotations

import logging
import queue
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import torch
from torch.utils.data import DataLoader

from .active_loop import ActiveLearningLoop
from ml.data_manager import ALDataManager
from ml.dataloader import get_datasets
from .events import Event, EventType, Inbox
from ml.models import get_model
from ml.strategies import get_strategy
from ml.trainer import Trainer

logger = logging.getLogger(__name__)


def build_al_loop(config: Any, run_dir: Path) -> ActiveLearningLoop:
    """Build the AL loop once per run."""
    exp_dir = Path(run_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if config.data.test_dir:
        from ml.dataloader import get_datasets_presplit
        datasets = get_datasets_presplit(
            train_dir=config.data.data_dir,
            test_dir=config.data.test_dir,
            val_split=config.data.val_split,
            augmentation=config.data.augmentation,
            seed=config.experiment.seed,
        )
    else:
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
        stratified_init=config.active_learning.stratified_init,
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
    event_inbox: Inbox,
    event_type: EventType,
    run_id: str,
    cycle: int = 0,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    event_inbox.put(
        Event(
            type=event_type,
            run_id=run_id,
            cycle=cycle,
            data=data or {},
        )
    )


def _check_stop(command_queue: "queue.Queue[Any]") -> bool:
    """Non-blocking check: drain the queue and return True if a STOP command is found.

    Non-STOP messages are preserved by requeueing them in original order so that
    later blocking waits (next-step, annotations) can still consume them.
    """
    buf: List[Any] = []
    found = False
    while True:
        try:
            msg = command_queue.get_nowait()
        except queue.Empty:
            break
        if msg.get("command") == "STOP":
            found = True
            break
        buf.append(msg)
    for msg in buf:
        command_queue.put(msg)
    return found


def _wait_for_next_step(command_queue: "queue.Queue[Any]") -> bool:
    """Block until NEXT_STEP or STOP command arrives. Returns False on STOP."""
    while True:
        try:
            msg = command_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        cmd = msg.get("command")
        if cmd == "STOP":
            return False
        if cmd == "NEXT_STEP":
            return True
        command_queue.put(msg)  # unknown command — requeue


def _wait_for_annotations(
    command_queue: "queue.Queue[Any]",
    run_id: str,
    cycle: int,
) -> Optional[List[Dict[str, Any]]]:
    """Block until SUBMIT_ANNOTATIONS (matching run/cycle) or STOP arrives.

    Returns the annotation list on success, or None if STOP was received.
    Mismatched SUBMIT_ANNOTATIONS commands (wrong run/cycle) are silently discarded.
    """
    while True:
        try:
            msg = command_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        cmd = msg.get("command")
        if cmd == "STOP":
            return None
        if cmd == "SUBMIT_ANNOTATIONS":
            if msg.get("run_id") == run_id and msg.get("cycle") == cycle:
                return list(msg.get("annotations", []))
            # Wrong run/cycle — discard silently
            continue
        command_queue.put(msg)  # unknown command — requeue


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


def _flush_artifacts(al_loop: Optional[ActiveLearningLoop], run_dir: Path) -> None:
    """Best-effort save of all run artifacts."""
    if al_loop is None:
        return
    try:
        al_loop.persist_artifacts()
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to flush artifacts")


def _persist_incremental_artifacts(al_loop: ActiveLearningLoop, cycle: int) -> float:
    """Best-effort persistence after each cycle so UI can read run folders directly."""
    start_ts = time.perf_counter()
    try:
        al_loop.persist_artifacts()
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to persist incremental artifacts for cycle %s", cycle)
    return time.perf_counter() - start_ts


def _exit_stopped(
    event_inbox: Inbox,
    run_id: str,
    cycle: int,
    al_loop: Optional[ActiveLearningLoop],
    run_dir: Path,
) -> None:
    _flush_artifacts(al_loop, run_dir)
    _emit_event(event_inbox, EventType.RUN_STOPPED, run_id=run_id, cycle=cycle, data={})


def run_experiment(
    command_queue: "queue.Queue[Any]",
    event_inbox: Inbox,
    config: Any,
    run_dir: Path,
    run_id: str,
) -> None:
    """
    Entry point for the backend daemon thread.

    Receives control signals via *command_queue* (STOP, NEXT_STEP, SUBMIT_ANNOTATIONS)
    and emits progress events via *event_inbox* (worker → UI direction).
    """
    al_loop: Optional[ActiveLearningLoop] = None
    current_cycle = 0

    try:
        al_loop = build_al_loop(config, run_dir)

        if _check_stop(command_queue):
            _exit_stopped(event_inbox, run_id, current_cycle, al_loop, run_dir)
            return

        total_cycles = int(config.active_learning.num_cycles)
        auto_annotate = bool(config.active_learning.auto_annotate)
        step_mode = bool(getattr(config.active_learning, "step_mode", False))

        for cycle in range(1, total_cycles + 1):
            if step_mode and cycle > 1:
                _emit_event(
                    event_inbox,
                    EventType.WAITING_FOR_STEP,
                    run_id=run_id,
                    cycle=cycle,
                    data={"next_cycle": cycle, "total_cycles": total_cycles},
                )
                if not _wait_for_next_step(command_queue):
                    _exit_stopped(event_inbox, run_id, current_cycle, al_loop, run_dir)
                    return

            current_cycle = cycle
            if _check_stop(command_queue):
                _exit_stopped(event_inbox, run_id, cycle, al_loop, run_dir)
                return

            prep_start = time.perf_counter()
            prep_info = al_loop.prepare_cycle(cycle)
            prep_elapsed = time.perf_counter() - prep_start
            logger.info("Cycle %s timing | prepare_cycle=%.2fs", cycle, prep_elapsed)
            pool_stats = _pool_stats(al_loop)
            class_names = list(getattr(al_loop, "class_names", []))
            _emit_event(
                event_inbox,
                EventType.CYCLE_STARTED,
                run_id=run_id,
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
                if _check_stop(command_queue):
                    _exit_stopped(event_inbox, run_id, cycle, al_loop, run_dir)
                    return

                metrics = al_loop.train_single_epoch(epoch)
                metrics_dict = _as_dict(metrics)
                if not metrics_dict:
                    metrics_dict = {"epoch": epoch}
                _emit_event(
                    event_inbox,
                    EventType.EPOCH_DONE,
                    run_id=run_id,
                    cycle=cycle,
                    data={
                        "epoch": epoch,
                        "total_epochs": epochs,
                        "metrics": metrics_dict,
                    },
                )

                if _check_stop(command_queue):
                    _exit_stopped(event_inbox, run_id, cycle, al_loop, run_dir)
                    return

                if al_loop.should_stop_early():
                    break

            if _check_stop(command_queue):
                _exit_stopped(event_inbox, run_id, cycle, al_loop, run_dir)
                return

            al_loop.trainer.restore_best_model()
            test_metrics = al_loop.run_evaluation()
            cycle_metrics = al_loop.finalize_cycle(test_metrics).model_dump()
            probe_images = _serialize_probe_images(al_loop)
            pool_stats = _pool_stats(al_loop)
            _emit_event(
                event_inbox,
                EventType.EVAL_COMPLETE,
                run_id=run_id,
                cycle=cycle,
                data={
                    "cycle_metrics": cycle_metrics,
                    "probe_images": probe_images,
                    "labeled_class_distribution": pool_stats["labeled_class_distribution"],
                    "unlabeled_class_distribution": pool_stats["unlabeled_class_distribution"],
                },
            )
            persist_elapsed = _persist_incremental_artifacts(al_loop, cycle)
            logger.info("Cycle %s timing | post_eval_persist=%.2fs", cycle, persist_elapsed)

            if cycle >= total_cycles:
                continue

            if int(cycle_metrics.get("unlabeled_pool_size", 0)) <= 0:
                break

            if _check_stop(command_queue):
                _exit_stopped(event_inbox, run_id, cycle, al_loop, run_dir)
                return

            _emit_event(
                event_inbox,
                EventType.QUERYING_STARTED,
                run_id=run_id,
                cycle=cycle,
                data={"cycle": cycle},
            )

            if auto_annotate:
                if _check_stop(command_queue):
                    _exit_stopped(event_inbox, run_id, cycle, al_loop, run_dir)
                    return

                auto_annotate_start = time.perf_counter()
                summary = al_loop.query_and_auto_annotate(heartbeat_fn=lambda: None)
                auto_annotate_elapsed = time.perf_counter() - auto_annotate_start
                logger.info(
                    "Cycle %s timing | auto_query_apply=%.2fs | queried=%s applied=%s",
                    cycle,
                    auto_annotate_elapsed,
                    int(summary.get("queried_count", 0)),
                    int(summary.get("applied_count", 0)),
                )
                if int(summary.get("queried_count", 0)) <= 0:
                    continue

                pool_stats = _pool_stats(al_loop)
                _emit_event(
                    event_inbox,
                    EventType.ANNOTATIONS_APPLIED,
                    run_id=run_id,
                    cycle=cycle,
                    data={
                        "count": int(summary.get("applied_count", 0)),
                        "labeled_pool_size": pool_stats["labeled_pool_size"],
                        "unlabeled_pool_size": pool_stats["unlabeled_pool_size"],
                        "labeled_class_distribution": pool_stats["labeled_class_distribution"],
                        "unlabeled_class_distribution": pool_stats["unlabeled_class_distribution"],
                    },
                )
                persist_elapsed = _persist_incremental_artifacts(al_loop, cycle)
                logger.info("Cycle %s timing | post_annotation_persist=%.2fs", cycle, persist_elapsed)
                continue

            query_start = time.perf_counter()
            queried_images = al_loop.query_samples(heartbeat_fn=lambda: None)
            query_elapsed = time.perf_counter() - query_start
            logger.info(
                "Cycle %s timing | manual_query_build_payload=%.2fs | queried=%s",
                cycle,
                query_elapsed,
                len(queried_images),
            )
            queried_dicts = [img.to_dict() for img in queried_images]

            if not queried_dicts:
                continue

            _emit_event(
                event_inbox,
                EventType.NEW_IMAGES,
                run_id=run_id,
                cycle=cycle,
                data={
                    "queried_images": queried_dicts,
                    "query_token": str(uuid4()),
                },
            )

            annotations = _wait_for_annotations(command_queue, run_id, cycle)
            if annotations is None:
                _exit_stopped(event_inbox, run_id, cycle, al_loop, run_dir)
                return

            if _check_stop(command_queue):
                _exit_stopped(event_inbox, run_id, cycle, al_loop, run_dir)
                return

            apply_start = time.perf_counter()
            al_loop.receive_annotations(annotations)
            apply_elapsed = time.perf_counter() - apply_start
            logger.info(
                "Cycle %s timing | manual_annotation_apply=%.2fs | count=%s",
                cycle,
                apply_elapsed,
                len(annotations),
            )
            pool_stats = _pool_stats(al_loop)
            _emit_event(
                event_inbox,
                EventType.ANNOTATIONS_APPLIED,
                run_id=run_id,
                cycle=cycle,
                data={
                    "count": len(annotations),
                    "labeled_pool_size": pool_stats["labeled_pool_size"],
                    "unlabeled_pool_size": pool_stats["unlabeled_pool_size"],
                    "labeled_class_distribution": pool_stats["labeled_class_distribution"],
                    "unlabeled_class_distribution": pool_stats["unlabeled_class_distribution"],
                },
            )
            persist_elapsed = _persist_incremental_artifacts(al_loop, cycle)
            logger.info("Cycle %s timing | post_annotation_persist=%.2fs", cycle, persist_elapsed)

        _flush_artifacts(al_loop, run_dir)
        _emit_event(
            event_inbox,
            EventType.RUN_FINISHED,
            run_id=run_id,
            cycle=current_cycle,
            data={},
        )

    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Backend thread failed")
        _flush_artifacts(al_loop, run_dir)
        _emit_event(
            event_inbox,
            EventType.RUN_ERROR,
            run_id=run_id,
            cycle=current_cycle,
            data={
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
