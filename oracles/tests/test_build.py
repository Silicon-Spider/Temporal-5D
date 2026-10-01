import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle_harness import build, lockfile, paths
from oracle_harness.util import HarnessError

# A made-up module in the shape cwmtt_layouts() has to read: a `layouts` table of
# (name, value) pairs, with other tables around it that must not be mistaken for it.
LAYOUTS_HS = '''
module Example.Layouts where

aliases =
  [ ("Not a layout", undefined)
  ]

layouts =
  [ ("Standard",plainStart)
  , ("Standard - Turn Zero", withTurnZero plainStart)
  , ("Focused - Just Kings", fromPieces "kings only")
  ]

pick :: String -> Start
pick name = fromMaybe plainStart (lookup name layouts)
'''


class CwmttLayoutTests(unittest.TestCase):
    def test_names_are_read_from_the_layouts_table_only(self):
        self.assertEqual(build.cwmtt_layouts(LAYOUTS_HS), ["Standard", "Standard - Turn Zero", "Focused - Just Kings"])

    def test_an_unrecognisable_table_stops_the_build(self):
        with self.assertRaises(HarnessError):
            build.cwmtt_layouts("module Layouts where\n")
        with self.assertRaises(HarnessError):
            build.cwmtt_layouts('layouts =\n  [ ("Somewhere else", x)\n  ]\n')


class BuildKeyTests(unittest.TestCase):
    def setUp(self):
        self.lock = lockfile.load()

    def test_the_key_follows_the_pin(self):
        import dataclasses

        pin = self.lock.pin("cwmtt")
        moved = dataclasses.replace(pin, commit="0" * 40)
        self.assertEqual(build.build_key(pin), build.build_key(pin))
        self.assertNotEqual(build.build_key(pin), build.build_key(moved))

    def test_the_key_follows_the_driver_but_not_its_line_endings(self):
        pin = self.lock.pin("ftxi")
        original = build.build_key(pin)
        with tempfile.TemporaryDirectory() as folder:
            drivers = Path(folder)
            (drivers / "ftxi").mkdir()
            for name in ("CMakeLists.txt", "ftxi_driver.cpp"):
                data = (paths.DRIVERS / "ftxi" / name).read_bytes().replace(b"\r\n", b"\n")
                (drivers / "ftxi" / name).write_bytes(data.replace(b"\n", b"\r\n"))
            with mock.patch.object(paths, "DRIVERS", drivers):
                self.assertEqual(build.build_key(pin), original)
                with open(drivers / "ftxi" / "ftxi_driver.cpp", "ab") as handle:
                    handle.write(b"// edited\n")
                self.assertNotEqual(build.build_key(pin), original)

    def test_every_oracle_has_a_recipe(self):
        self.assertEqual(set(build.buildable(self.lock)), {pin.id for pin in self.lock.oracles()})

    def test_a_finished_build_replaces_the_stale_ones_of_the_same_oracle(self):
        with tempfile.TemporaryDirectory() as folder:
            oracle = Path(folder) / "build" / "ftxi"
            other = Path(folder) / "build" / "cwmtt" / "cccc"
            for path in (oracle / "aaaa" / "src", oracle / "bbbb" / "src", other):
                path.mkdir(parents=True)
            build.prune_siblings(oracle / "bbbb")
            self.assertEqual(sorted(path.name for path in oracle.iterdir()), ["bbbb"])
            self.assertTrue(other.exists())


class ToolsNeededTests(unittest.TestCase):
    """`build --plan`: what CI must install before `build` (only the tools of what is not built)."""

    def setUp(self):
        self.lock = lockfile.load()

    def test_every_recipe_names_its_tools(self):
        for oracle in build.buildable(self.lock):
            self.assertTrue(build.recipe_tools(oracle), oracle)

    def test_an_up_to_date_build_needs_no_tools(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {"ORACLE_HARNESS_CACHE": folder}):
            everything = build.buildable(self.lock)
            self.assertIn("ghc", build.tools_needed(self.lock, everything))
            pin = self.lock.pin("cwmtt")
            root = build.build_root(pin)
            root.mkdir(parents=True)
            (root / "cwmtt").write_bytes(b"")
            build._finish(pin, root, {"cli": "cwmtt"}, {})
            self.assertNotIn("ghc", build.tools_needed(self.lock, everything))
            self.assertEqual(build.tools_needed(self.lock, ["cwmtt"]), [])
            self.assertEqual(build.tools_needed(self.lock, ["ftxi"]), sorted(build.recipe_tools("ftxi")))


if __name__ == "__main__":
    unittest.main()
