import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from controller import Controller
from experiment_state import AppState


def make_config(num_cycles: int = 3):
    return SimpleNamespace(
        active_learning=SimpleNamespace(num_cycles=num_cycles, auto_annotate=False),
        training=SimpleNamespace(epochs=1),
        data=SimpleNamespace(num_workers=4),
    )


class ControllerLifecycleTests(unittest.TestCase):
    def test_stop_sets_stopping_state_immediately(self):
        config = make_config()
        controller = Controller(config)

        def fake_run(state, _config):
            run_id = state.snapshot()["run_id"]
            state.update_for_run(run_id, thread_status="running")
            while not state.stop_event.is_set():
                time.sleep(0.01)
            # Intentionally sleep to keep thread alive past a short join timeout.
            time.sleep(0.2)

        with patch("controller.run_experiment", side_effect=fake_run):
            controller.start_experiment(config)
            time.sleep(0.03)
            controller.stop_experiment(join_timeout=0.01)
            snap = controller.get_snapshot()
            self.assertEqual(snap["app_state"], AppState.STOPPING)
            self.assertIn(snap["thread_status"], {"stopping", "running"})

            # Allow thread to exit to avoid cross-test interference.
            time.sleep(0.25)

    def test_start_stop_start_prevents_old_run_writes(self):
        config = make_config()
        controller = Controller(config)
        call_counter = {"count": 0}

        def fake_run(state, _config):
            call_counter["count"] += 1
            this_call = call_counter["count"]
            run_id = state.snapshot()["run_id"]
            if this_call == 1:
                # Old thread lingers and attempts a stale write after run switch.
                time.sleep(1.2)
                state.update_for_run(run_id, current_cycle=999, progress_detail="stale")
            else:
                # New thread exits quickly.
                time.sleep(0.1)

        with patch("controller.run_experiment", side_effect=fake_run):
            first_run = controller.start_experiment(config)
            time.sleep(0.05)
            second_run = controller.start_experiment(config)
            self.assertNotEqual(first_run, second_run)

            # Wait until old thread attempted its stale write.
            time.sleep(1.35)
            snap = controller.get_snapshot()
            self.assertEqual(snap["run_id"], second_run)
            self.assertNotEqual(snap["current_cycle"], 999)

            controller.stop_experiment(join_timeout=0.2)


if __name__ == "__main__":
    unittest.main()
