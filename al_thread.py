"""
Background thread orchestration for interactive active learning.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from active_loop import ActiveLearningLoop
from data_manager import ALDataManager
from dataloader import get_datasets
from experiment_state import AppState, ExperimentState
from models import get_model
from strategies import get_strategy
from trainer import Trainer

logger = logging.getLogger(__name__)


def build_al_loop(config: Any) -> ActiveLearningLoop:
    """Build the AL loop once per run."""
    exp_dir = Path(config.experiment.exp_dir) / config.experiment.name
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


def _should_abort(state: ExperimentState, run_id: str) -> bool:
    if state.stop_event.is_set():
        return True
    return not state.is_run_active(run_id)


def _exit_stopped(state: ExperimentState, run_id: str) -> None:
    if state.is_run_active(run_id):
        state.update_for_run(
            run_id,
            app_state=AppState.IDLE,
            thread_status="finished",
            progress_detail="Stopped",
        )


def _append_epoch_metric(state: ExperimentState, run_id: str, metric: Dict[str, Any]) -> bool:
    snap = state.snapshot()
    metrics = snap["epoch_metrics"]
    metrics.append(metric)
    return state.update_for_run(run_id, epoch_metrics=metrics, current_epoch=metric.get("epoch", 0))


def _append_cycle_metric(state: ExperimentState, run_id: str, metric: Dict[str, Any]) -> bool:
    snap = state.snapshot()
    history = snap["metrics_history"]
    history.append(metric)
    return state.update_for_run(
        run_id,
        metrics_history=history,
        unlabeled_pool_size=int(metric.get("unlabeled_pool_size", 0)),
    )


def run_experiment(state: ExperimentState, config: Any) -> None:
    """
    Entry point for the backend daemon thread.
    """
    local_run_id = state.snapshot()["run_id"]

    try:
        if not state.update_for_run(
            local_run_id,
            app_state=AppState.INITIALIZING,
            thread_status="running",
            progress_detail="Building model and dataloaders...",
        ):
            return
        state.add_event("Backend thread started", run_id=local_run_id)

        al_loop = build_al_loop(config)
        state.add_event("Active learning loop initialized", run_id=local_run_id)
        state.touch_heartbeat(local_run_id)

        if _should_abort(state, local_run_id):
            _exit_stopped(state, local_run_id)
            return

        total_cycles = int(config.active_learning.num_cycles)
        auto_annotate = bool(config.active_learning.auto_annotate)

        for cycle in range(1, total_cycles + 1):
            if _should_abort(state, local_run_id):
                _exit_stopped(state, local_run_id)
                return

            if not state.update_for_run(
                local_run_id,
                app_state=AppState.TRAINING,
                current_cycle=cycle,
                current_epoch=0,
                epoch_metrics=[],
                progress_detail=f"Preparing cycle {cycle}/{total_cycles}...",
            ):
                return
            state.add_event("Cycle started", run_id=local_run_id, cycle=cycle, total_cycles=total_cycles)

            prep_info = al_loop.prepare_cycle(cycle)
            class_names = list(getattr(al_loop, "class_names", []))
            if not state.update_for_run(
                local_run_id,
                class_names=class_names,
                unlabeled_pool_size=int(prep_info.get("unlabeled_count", 0)),
                progress_detail=f"Training cycle {cycle}/{total_cycles}...",
            ):
                return
            state.add_event(
                "Cycle prepared",
                run_id=local_run_id,
                cycle=cycle,
                labeled=int(prep_info.get("labeled_count", 0)),
                unlabeled=int(prep_info.get("unlabeled_count", 0)),
            )

            epochs = int(config.training.epochs)
            for epoch in range(1, epochs + 1):
                if _should_abort(state, local_run_id):
                    _exit_stopped(state, local_run_id)
                    return

                state.update_for_run(
                    local_run_id,
                    app_state=AppState.TRAINING,
                    current_epoch=epoch,
                    progress_detail=f"Cycle {cycle}/{total_cycles} - Epoch {epoch}/{epochs}",
                )
                state.add_event("Epoch started", run_id=local_run_id, cycle=cycle, epoch=epoch)
                state.touch_heartbeat(local_run_id)

                metrics = al_loop.train_single_epoch(epoch)
                metrics_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics)
                if not _append_epoch_metric(state, local_run_id, metrics_dict):
                    return
                state.add_event(
                    "Epoch finished",
                    run_id=local_run_id,
                    cycle=cycle,
                    epoch=epoch,
                    train_loss=float(metrics_dict.get("train_loss", 0.0)),
                    val_acc=float(metrics_dict.get("val_accuracy") or 0.0),
                )

                if _should_abort(state, local_run_id):
                    _exit_stopped(state, local_run_id)
                    return

                if al_loop.should_stop_early():
                    state.add_event("Early stopping triggered", run_id=local_run_id, cycle=cycle, epoch=epoch)
                    break

            if _should_abort(state, local_run_id):
                _exit_stopped(state, local_run_id)
                return

            state.update_for_run(
                local_run_id,
                app_state=AppState.TRAINING,
                progress_detail=f"Evaluating cycle {cycle}/{total_cycles}...",
            )
            state.add_event("Cycle evaluation started", run_id=local_run_id, cycle=cycle)
            state.touch_heartbeat(local_run_id)

            test_metrics = al_loop.run_evaluation()
            cycle_metrics = al_loop.finalize_cycle(test_metrics).model_dump()
            state.add_event(
                "Cycle evaluation finished",
                run_id=local_run_id,
                cycle=cycle,
                test_acc=float(cycle_metrics.get("test_accuracy", 0.0)),
            )

            if not _append_cycle_metric(state, local_run_id, cycle_metrics):
                return

            state.update_for_run(
                local_run_id,
                queried_images=[],
                progress_detail=f"Cycle {cycle}/{total_cycles} complete",
            )
            state.add_event(
                "Cycle finalized",
                run_id=local_run_id,
                cycle=cycle,
                unlabeled=int(cycle_metrics.get("unlabeled_pool_size", 0)),
            )

            if cycle >= total_cycles:
                continue

            if int(cycle_metrics.get("unlabeled_pool_size", 0)) <= 0:
                state.add_event("Stopping early: unlabeled pool exhausted", run_id=local_run_id, cycle=cycle)
                break

            if _should_abort(state, local_run_id):
                _exit_stopped(state, local_run_id)
                return

            state.update_for_run(
                local_run_id,
                app_state=AppState.QUERYING,
                progress_detail=f"Querying samples for cycle {cycle}/{total_cycles}...",
            )
            state.add_event("Query started", run_id=local_run_id, cycle=cycle)
            state.touch_heartbeat(local_run_id)

            queried_images = al_loop.query_samples()
            queried_dicts = [img.to_dict() for img in queried_images]
            if not state.update_for_run(local_run_id, queried_images=queried_dicts):
                return
            state.add_event("Query finished", run_id=local_run_id, cycle=cycle, queried=len(queried_dicts))

            if not queried_dicts:
                state.add_event("No samples returned by query strategy", run_id=local_run_id, cycle=cycle)
                continue

            if auto_annotate:
                if _should_abort(state, local_run_id):
                    _exit_stopped(state, local_run_id)
                    return

                state.update_for_run(
                    local_run_id,
                    app_state=AppState.QUERYING,
                    progress_detail=f"Auto-annotating {len(queried_dicts)} samples...",
                )

                annotations = [
                    {"image_id": img["image_id"], "user_label": img["ground_truth"]}
                    for img in queried_dicts
                ]
                al_loop.receive_annotations(annotations)
                state.update_for_run(
                    local_run_id,
                    progress_detail=f"Auto-annotation complete for cycle {cycle}/{total_cycles}",
                    queried_images=[],
                )
                state.add_event("Auto-annotation applied", run_id=local_run_id, cycle=cycle, count=len(annotations))
                state.clear_annotations()
                continue

            state.clear_annotations()
            if not state.update_for_run(
                local_run_id,
                app_state=AppState.ANNOTATING,
                progress_detail=f"Waiting for annotations (cycle {cycle}/{total_cycles})...",
            ):
                return
            state.add_event("Waiting for manual annotations", run_id=local_run_id, cycle=cycle, required=len(queried_dicts))

            while True:
                if _should_abort(state, local_run_id):
                    _exit_stopped(state, local_run_id)
                    return
                if state.annotations_ready.wait(timeout=0.5):
                    annotations = state.consume_annotations(local_run_id, cycle)
                    if annotations is None:
                        state.add_event("Annotation signal ignored due run/cycle mismatch", run_id=local_run_id, cycle=cycle)
                        continue
                    break
                state.touch_heartbeat(local_run_id)

            if _should_abort(state, local_run_id):
                _exit_stopped(state, local_run_id)
                return

            state.update_for_run(
                local_run_id,
                app_state=AppState.ANNOTATING,
                progress_detail=f"Applying {len(annotations)} annotations...",
            )
            state.add_event("Manual annotations received", run_id=local_run_id, cycle=cycle, count=len(annotations))
            al_loop.receive_annotations(annotations)
            state.update_for_run(
                local_run_id,
                queried_images=[],
                progress_detail=f"Annotations applied for cycle {cycle}/{total_cycles}",
            )
            state.add_event("Manual annotations applied", run_id=local_run_id, cycle=cycle, count=len(annotations))

        if state.is_run_active(local_run_id):
            state.update_for_run(
                local_run_id,
                app_state=AppState.FINISHED,
                thread_status="finished",
                progress_detail="Experiment finished",
            )
            state.add_event("Run finished", run_id=local_run_id)

    except Exception as exc:  # pylint: disable=broad-exception-caught
        if state.is_run_active(local_run_id):
            tb = traceback.format_exc()
            logger.exception("Backend thread failed")
            state.set_error(exc, tb)
