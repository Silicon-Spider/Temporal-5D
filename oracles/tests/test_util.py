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


# Prints a line, waits, prints another: the second must be seen to arrive later.
PACED = "import sys, time\nprint('first', flush=True)\ntime.sleep(1.5)\nprint('second', flush=True)\n"
# Prints a line a second, forever.
TICKING = "import time\nfor n in range(1000):\n    print(f'tick {n}', flush=True)\n    time.sleep(0.2)\n"


class ArrivalTests(unittest.TestCase):
    """When each line arrived is the cost of an answer (H32): the output is read as it comes."""

    def test_each_line_is_timed_as_it_arrives(self):
        done = util.run([sys.executable, "-c", PACED], timeout=30)
        self.assertEqual(done.out_text().split(), ["first", "second"])
        first = done.arrived(done.stdout.index(b"first"))
        second = done.arrived(done.stdout.index(b"second"))
        self.assertIsNotNone(first)
        self.assertGreater(second - first, 1.0)
        self.assertGreaterEqual(done.elapsed_s, second)
        self.assertIsNone(done.arrived(len(done.stdout)))  # never arrived
        # A stand-in with nothing measured knows nothing of when anything arrived.
        self.assertIsNone(util.Completed(0, b"x\n", b"", False).arrived(0))

    def test_a_stop_condition_ends_the_command_once_it_has_what_it_needs(self):
        seen = bytearray()

        def third_tick(chunk: bytes) -> bool:
            seen.extend(chunk)
            return b"tick 2" in seen

        started = time.monotonic()
        done = util.run([sys.executable, "-c", TICKING], timeout=60, stop=third_tick)
        self.assertLess(time.monotonic() - started, 30)
        self.assertTrue(done.stopped)
        self.assertFalse(done.timed_out)
        self.assertTrue(done.killed)
        self.assertIsNone(done.returncode)
        self.assertIn("tick 2", done.out_text())

    def test_a_broken_stop_condition_never_stops_a_command(self):
        def broken(chunk: bytes) -> bool:
            raise RuntimeError("a bug in the condition")

        done = util.run([sys.executable, "-c", PACED], timeout=30, stop=broken)
        self.assertFalse(done.stopped)
        self.assertEqual((done.returncode, done.out_text().split()), (0, ["first", "second"]))

    def test_a_time_out_is_not_a_stop(self):
        done = util.run([sys.executable, "-c", TICKING], timeout=1, stop=lambda chunk: False)
        self.assertEqual((done.timed_out, done.stopped), (True, False))
        self.assertIn("tick 0", done.out_text())


if __name__ == "__main__":
    unittest.main()
