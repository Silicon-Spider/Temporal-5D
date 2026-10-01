import sys
import time
import unittest

from oracle_harness import util

# A command that starts a child and waits for it. The child inherits the
# command's output pipes and would hold them open for a minute.
WRAPPER = (
    "import subprocess, sys, time\n"
    "print('started', flush=True)\n"
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "time.sleep(60)\n"
)


class RunTests(unittest.TestCase):
    def test_output_and_exit_code(self):
        done = util.run([sys.executable, "-c", "import sys; print(sys.stdin.read().upper()); sys.exit(3)"], stdin=b"abc")
        self.assertEqual((done.returncode, done.out_text().strip(), done.timed_out), (3, "ABC", False))

    def test_a_command_that_runs_out_of_time_keeps_what_it_printed(self):
        done = util.run([sys.executable, "-c", "import time; print('one', flush=True); time.sleep(60)"], timeout=3)
        self.assertTrue(done.timed_out)
        self.assertIsNone(done.returncode)
        self.assertEqual(done.out_text().strip(), "one")

    def test_a_time_out_takes_the_commands_children_with_it(self):
        # Killing only the wrapper would leave the child holding the pipes, and
        # the gate would sit there until the child chose to exit.
        started = time.monotonic()
        done = util.run([sys.executable, "-c", WRAPPER], timeout=3)
        elapsed = time.monotonic() - started
        self.assertTrue(done.timed_out)
        self.assertEqual(done.out_text().strip(), "started")
        self.assertLess(elapsed, 30, "the run waited for a grandchild it should have killed")

    def test_nothing_is_left_registered_as_running(self):
        util.run([sys.executable, "-c", "pass"])
        util.kill_running()  # nothing to kill, and it must not raise
        self.assertEqual(len(util._running), 0)

    def test_check_run_reports_a_failure_with_what_the_command_said(self):
        with self.assertRaises(util.HarnessError) as raised:
            util.check_run([sys.executable, "-c", "import sys; sys.stderr.write('no such pin'); sys.exit(2)"])
        self.assertIn("exited with 2", str(raised.exception))
        self.assertIn("no such pin", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
