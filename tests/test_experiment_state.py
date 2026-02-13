import threading
import time
import unittest
from types import SimpleNamespace

from experiment_state import AppState, ExperimentState


def make_config(num_cycles: int = 3):
    return SimpleNamespace(active_learning=SimpleNamespace(num_cycles=num_cycles))


class ExperimentStateTests(unittest.TestCase):
    def test_snapshot_is_atomic_for_related_fields(self):
        state = ExperimentState()
        run_id = state.reset(make_config())
        state.update_for_run(run_id, app_state=AppState.TRAINING)

        stop = threading.Event()

        def writer():
            n = 1
            while not stop.is_set():
                state.update_for_run(run_id, current_cycle=n, current_epoch=n)
                n += 1

        t = threading.Thread(target=writer, daemon=True)
        t.start()

        for _ in range(2000):
            snap = state.snapshot()
            self.assertEqual(snap["current_cycle"], snap["current_epoch"])

        stop.set()
        t.join(timeout=1.0)

    def test_snapshot_returns_copied_collections(self):
        state = ExperimentState()
        run_id = state.reset(make_config())
        state.update_for_run(run_id, epoch_metrics=[{"epoch": 1}], metrics_history=[{"cycle": 1}])

        snap = state.snapshot()
        snap["epoch_metrics"].append({"epoch": 99})
        snap["metrics_history"].append({"cycle": 99})

        snap2 = state.snapshot()
        self.assertEqual(len(snap2["epoch_metrics"]), 1)
        self.assertEqual(len(snap2["metrics_history"]), 1)

    def test_set_annotations_rejects_stale_run_or_cycle(self):
        state = ExperimentState()
        run_id = state.reset(make_config())
        state.update_for_run(run_id, app_state=AppState.ANNOTATING, current_cycle=2)

        ok = state.set_annotations(run_id, 2, [{"image_id": 1, "user_label": 0}])
        self.assertTrue(ok)

        bad_run = state.set_annotations("other-run", 2, [{"image_id": 1, "user_label": 0}])
        self.assertFalse(bad_run)

        bad_cycle = state.set_annotations(run_id, 3, [{"image_id": 1, "user_label": 0}])
        self.assertFalse(bad_cycle)

        state.update_for_run(run_id, app_state=AppState.TRAINING)
        bad_state = state.set_annotations(run_id, 2, [{"image_id": 1, "user_label": 0}])
        self.assertFalse(bad_state)


if __name__ == "__main__":
    unittest.main()
