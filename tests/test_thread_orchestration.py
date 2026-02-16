import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from al_thread import run_experiment
from experiment_state import AppState, ExperimentState


def make_config(num_cycles: int = 2, auto_annotate: bool = False, epochs: int = 1):
    return SimpleNamespace(
        active_learning=SimpleNamespace(num_cycles=num_cycles, auto_annotate=auto_annotate),
        training=SimpleNamespace(epochs=epochs),
    )


class DummyEpochMetrics:
    def __init__(self, epoch: int):
        self.epoch = epoch

    def to_dict(self):
        return {
            "epoch": self.epoch,
            "train_loss": 0.1,
            "train_accuracy": 0.8,
            "val_loss": 0.2,
            "val_accuracy": 0.75,
            "learning_rate": 1e-4,
        }


class DummyCycleMetrics:
    def __init__(self, cycle: int, unlabeled_pool_size: int):
        self.cycle = cycle
        self.unlabeled_pool_size = unlabeled_pool_size

    def model_dump(self):
        return {
            "cycle": self.cycle,
            "labeled_pool_size": 100 + self.cycle,
            "unlabeled_pool_size": self.unlabeled_pool_size,
            "epochs_trained": 1,
            "best_val_accuracy": 0.75,
            "best_epoch": 1,
            "test_accuracy": 0.8,
            "test_f1": 0.79,
            "test_precision": 0.78,
            "test_recall": 0.81,
        }


class DummyQueriedImage:
    def __init__(self, image_id: int, ground_truth: int):
        self.image_id = image_id
        self.ground_truth = ground_truth

    def to_dict(self):
        return {
            "image_id": self.image_id,
            "image_path": f"img_{self.image_id}.jpg",
            "display_path": f"img_{self.image_id}.jpg",
            "ground_truth": self.ground_truth,
            "ground_truth_name": f"Class_{self.ground_truth}",
            "model_probabilities": {"Class_0": 0.5, "Class_1": 0.5},
            "predicted_class": "Class_0",
            "predicted_confidence": 0.5,
            "uncertainty_score": 0.8,
            "selection_reason": "High entropy",
        }


class DummyLoop:
    def __init__(self):
        self.class_names = ["Class_0", "Class_1"]
        self.last_cycle = 0
        self.received_annotations = []

    def prepare_cycle(self, cycle: int):
        self.last_cycle = cycle
        return {"cycle": cycle, "unlabeled_count": 20}

    def train_single_epoch(self, epoch: int):
        return DummyEpochMetrics(epoch)

    def should_stop_early(self):
        return False

    def run_evaluation(self):
        return {"test_accuracy": 0.8}

    def finalize_cycle(self, _test_metrics):
        unlabeled = 20 if self.last_cycle == 1 else 0
        return DummyCycleMetrics(cycle=self.last_cycle, unlabeled_pool_size=unlabeled)

    def query_samples(self):
        return [DummyQueriedImage(1, 0), DummyQueriedImage(2, 1)]

    def receive_annotations(self, annotations):
        self.received_annotations.append(annotations)
        return {"moved_count": len(annotations)}


class ThreadOrchestrationTests(unittest.TestCase):
    def test_manual_annotation_wait_path(self):
        config = make_config(num_cycles=2, auto_annotate=False)
        state = ExperimentState()
        run_id = state.reset(config)
        loop = DummyLoop()

        with patch("al_thread.build_al_loop", return_value=loop):
            t = threading.Thread(target=run_experiment, args=(state, config), daemon=True)
            t.start()

            deadline = time.time() + 2.0
            while time.time() < deadline:
                snap = state.snapshot()
                if snap["app_state"] == AppState.ANNOTATING:
                    break
                time.sleep(0.02)

            snap = state.snapshot()
            self.assertEqual(snap["app_state"], AppState.ANNOTATING)

            submitted = [{"image_id": 1, "user_label": 0}, {"image_id": 2, "user_label": 1}]
            accepted = state.set_annotations(run_id, cycle=1, annotations=submitted)
            self.assertTrue(accepted)

            t.join(timeout=3.0)
            self.assertFalse(t.is_alive())
            self.assertGreaterEqual(len(loop.received_annotations), 1)
            self.assertEqual(loop.received_annotations[0], submitted)
            self.assertEqual(state.snapshot()["app_state"], AppState.FINISHED)

    def test_auto_annotate_path(self):
        config = make_config(num_cycles=2, auto_annotate=True)
        state = ExperimentState()
        state.reset(config)
        loop = DummyLoop()

        with patch("al_thread.build_al_loop", return_value=loop):
            run_experiment(state, config)

        self.assertGreaterEqual(len(loop.received_annotations), 1)
        first_batch = loop.received_annotations[0]
        self.assertEqual(
            first_batch,
            [{"image_id": 1, "user_label": 0}, {"image_id": 2, "user_label": 1}],
        )
        self.assertEqual(state.snapshot()["app_state"], AppState.FINISHED)

    def test_exception_sets_error_state(self):
        config = make_config()
        state = ExperimentState()
        state.reset(config)

        with patch("al_thread.build_al_loop", side_effect=RuntimeError("boom")), patch(
            "al_thread.logger.exception"
        ):
            run_experiment(state, config)

        snap = state.snapshot()
        self.assertEqual(snap["app_state"], AppState.ERROR)
        self.assertEqual(snap["thread_status"], "failed")
        self.assertIn("boom", snap["last_error"]["message"])


if __name__ == "__main__":
    unittest.main()
