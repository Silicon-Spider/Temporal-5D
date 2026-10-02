"""Recorded-answer integrity (decision H33): MANIFEST, the harness held still while a gate runs, and an entry
point that runs nothing from the tree but source."""

import importlib.machinery
import importlib.util
import marshal
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle_harness import integrity, paths
from oracle_harness.util import sha256_text


class ManifestTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        patcher = mock.patch.object(paths, "GOLDEN_DIR", self.folder)
        patcher.start()
        self.addCleanup(patcher.stop)

    def put(self, name, text):
        (self.folder / name).write_text(text, encoding="utf-8", newline="\n")

    def codes(self):
        return [finding.code for finding in integrity.verify_manifest()]

    def test_a_manifest_vouches_for_every_file_beside_it(self):
        self.put("ftxi.json", "{}\n")
        self.put("ftxi.cost.json", "{}\n")
        integrity.write_manifest({path.name: sha256_text(path.read_bytes()) for path in integrity.recorded_files()})
        text = integrity.manifest_path().read_text(encoding="utf-8")
        # sha256sum's format, sorted by name: `sha256sum -c MANIFEST` checks it without the harness.
        self.assertEqual(text, f"{sha256_text(b'{}' + chr(10).encode())}  ftxi.cost.json\n"
                               f"{sha256_text(b'{}' + chr(10).encode())}  ftxi.json\n")
        self.assertEqual(self.codes(), [])
        self.assertEqual(integrity.read_manifest(), {"ftxi.cost.json": sha256_text(b"{}\n"), "ftxi.json": sha256_text(b"{}\n")})
        self.assertEqual(integrity.verified_bytes(self.folder / "ftxi.json", integrity.read_manifest()), b"{}\n")

    def test_an_edit_an_unlisted_file_and_a_missing_one_are_each_found(self):
        self.put("ftxi.json", "{}\n")
        self.put("notation.json", "{}\n")
        integrity.write_manifest({path.name: sha256_text(path.read_bytes()) for path in integrity.recorded_files()})
        self.put("ftxi.json", '{"edited": true}\n')
        self.put("planted.json", "{}\n")
        (self.folder / "notation.json").unlink()
        found = integrity.verify_manifest()
        self.assertEqual([finding.code for finding in found], ["golden-integrity"] * 3)
        text = " | ".join(finding.message for finding in found)
        for words in ("lists notation.json, which is not there", "planted.json is in oracles/golden but not in MANIFEST",
                      "ftxi.json does not match its hash in MANIFEST"):
            self.assertIn(words, text)
        with self.assertRaises(integrity.IntegrityError):
            integrity.verified_bytes(self.folder / "ftxi.json", integrity.read_manifest())
        with self.assertRaises(integrity.IntegrityError):
            integrity.verified_bytes(self.folder / "planted.json", integrity.read_manifest())

    def test_without_a_manifest_recorded_answers_are_not_vouched_for(self):
        self.assertEqual(self.codes(), [])  # nothing recorded, nothing to vouch for
        self.put("ftxi.json", "{}\n")
        self.assertEqual(self.codes(), ["golden-integrity"])
        self.assertIsNone(integrity.read_manifest())
        with self.assertRaises(integrity.IntegrityError) as raised:
            integrity.verified_bytes(self.folder / "ftxi.json", None)
        self.assertIn("golden --write", str(raised.exception))

    def test_a_malformed_manifest_is_refused(self):
        for text in ("not a hash line\n", f"{'a' * 64} ../escape.json\n", f"{'a' * 64}  x.json\n{'b' * 64}  x.json\n"):
            integrity.manifest_path().write_text(text, encoding="utf-8")
            with self.assertRaises(integrity.IntegrityError, msg=text):
                integrity.read_manifest()
            self.assertEqual(self.codes(), ["golden-integrity"], text)


class HarnessHashTests(unittest.TestCase):
    """Every harness file a gate run compares against or executes, hashed when it starts and when it ends."""

    def test_it_covers_what_a_run_compares_against_and_executes(self):
        names = set(integrity.hash_harness())
        harness = paths.HARNESS_REL
        for expected in (f"{harness}/oracle_harness/gate.py", f"{harness}/oracle_harness/adapters/base.py",
                         f"{harness}/drivers/ftxi/ftxi_driver.cpp", f"{harness}/oracles.lock.toml",
                         f"{harness}/corpus/manifest.toml"):
            self.assertIn(expected, names)
        # Wherever they live (a test may point the gate at copies of its own).
        for path in (paths.DIVERGENCES_FILE, paths.GATE_FILE, paths.DISAGREEMENTS_FILE, *integrity.recorded_files()):
            self.assertIn(integrity._name(path), names)
        self.assertFalse(any("__pycache__" in name or name.endswith(".pyc") for name in names))

    def test_a_change_an_addition_and_a_removal_are_each_a_finding(self):
        before = {"oracles/a": "1", "oracles/b": "2", "oracles/c": "3"}
        after = {"oracles/a": "1", "oracles/b": "9", "oracles/d": "4", "oracles/c": "missing"}
        found = integrity.changes(before, after)
        self.assertEqual([finding.code for finding in found], ["harness-modified"] * 3)
        self.assertEqual([finding.message.split(" while")[0] for finding in found],
                         ["oracles/b changed", "oracles/c disappeared", "oracles/d appeared"])
        self.assertEqual(integrity.changes(before, dict(before)), [])

    def test_every_file_under_the_harness_but_its_caches_and_the_pinned_oracles(self):
        # A module added while a run lasts fails it as surely as an edit: everything under oracles/ is hashed, but
        # the caches (whose bytecode harness.py never runs) and the pinned oracles.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "oracles"
            kept = ("harness.py", "oracle_harness/gate.py", "oracle_harness/planted.py", "tests/test_x.py",
                    "drivers/ftxi/driver.cpp", "pyproject.toml")
            skipped = (".cache/build/ftxi/driver.exe", ".ruff_cache/x", "third_party/ftxi/src/a.cpp",
                       "oracle_harness/__pycache__/gate.cpython-314.pyc")
            for relative in (*kept, *skipped):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
            with mock.patch.object(paths, "HARNESS_ROOT", root), \
                    mock.patch.object(paths, "THIRD_PARTY", root / "third_party"), \
                    mock.patch.object(paths, "REPO_ROOT", Path(folder)):
                names = {integrity._name(path) for path in integrity.harness_files()}
        for relative in kept:
            self.assertIn(f"oracles/{relative}", names)
        for relative in skipped:
            self.assertNotIn(f"oracles/{relative}", names)

    def test_it_follows_the_files_a_test_points_the_gate_at(self):
        with tempfile.TemporaryDirectory() as folder:
            divergences = Path(folder) / "known-divergences.toml"
            divergences.write_text("schema = 1\n", encoding="utf-8")
            with mock.patch.object(paths, "DIVERGENCES_FILE", divergences):
                before = integrity.hash_harness()
                divergences.write_text("schema = 1\n# an entry the engine added\n", encoding="utf-8")
                found = integrity.changes(before, integrity.hash_harness())
        self.assertEqual(len(found), 1)
        self.assertIn("known-divergences.toml changed while the gate ran", found[0].message)


class EntryPointTests(unittest.TestCase):
    """harness.py runs nothing from the tree but source: no planted bytecode, no module shadowing the standard library."""

    def test_neither_planted_bytecode_nor_a_module_named_like_the_standard_librarys_is_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copy(paths.HARNESS_ROOT / "harness.py", root / "harness.py")
            package = root / "oracle_harness"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            source = package / "verdict.py"
            source.write_text("def say():\n    return 'source'\n", encoding="utf-8")
            (package / "cli.py").write_text(
                "import colorsys\n"
                "from . import verdict\n"
                "def main():\n"
                "    print(verdict.say(), getattr(colorsys, 'PLANTED', 'stdlib'))\n"
                "    return 0\n", encoding="utf-8")
            # A module beside the entry point, named like a standard-library one (that no start-up imports).
            (root / "colorsys.py").write_text("PLANTED = 'planted'\n", encoding="utf-8")
            # Bytecode for verdict.py that says something else, with the header of the source's own cache file -
            # its time and size - which is all Python checks.
            stat = source.stat()
            header = (importlib.util.MAGIC_NUMBER + (0).to_bytes(4, "little")
                      + (int(stat.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
                      + (stat.st_size & 0xFFFFFFFF).to_bytes(4, "little"))
            code = compile("def say():\n    return 'planted bytecode'\n", str(source), "exec")
            cache = package / "__pycache__" / f"verdict.{sys.implementation.cache_tag}.pyc"
            cache.parent.mkdir()
            cache.write_bytes(header + marshal.dumps(code))
            env = {key: value for key, value in os.environ.items()
                   if key not in ("PYTHONPYCACHEPREFIX", "PYTHONDONTWRITEBYTECODE", "PYTHONSAFEPATH", "PYTHONPATH")}
            guarded = subprocess.run([sys.executable, str(root / "harness.py")], capture_output=True, text=True,
                                     env=env, cwd=root, check=False)
            self.assertEqual((guarded.returncode, guarded.stdout.strip()), (0, "source stdlib"), guarded.stderr)
            # And it wrote no bytecode into the tree.
            self.assertEqual(sorted(path.name for path in cache.parent.iterdir()), [cache.name])
            # Without the entry point's precautions Python runs both: the attack is real.
            plain = subprocess.run(
                [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import colorsys; "
                 "from oracle_harness import verdict; print(verdict.say(), getattr(colorsys, 'PLANTED', 'stdlib'))",
                 str(root)], capture_output=True, text=True, env=env, check=False)
            self.assertEqual(plain.stdout.strip(), "planted bytecode planted", plain.stderr)

    def test_an_extension_module_planted_beside_a_source_file_is_not_loaded(self):
        # Python's own finder tries an extension module (verdict.pyd, verdict.cpython-314-x86_64-linux-gnu.so) ahead
        # of the source file of the same name - and runs it on every later run, its source and hash unchanged.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copy(paths.HARNESS_ROOT / "harness.py", root / "harness.py")
            package = root / "oracle_harness"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "verdict.py").write_text("def say():\n    return 'source'\n", encoding="utf-8")
            (package / "cli.py").write_text("from . import verdict\ndef main():\n    print(verdict.say())\n    return 0\n",
                                            encoding="utf-8")
            # Not a real extension module - Python's finder fails on it, which shows that it would have loaded one.
            (package / f"verdict{importlib.machinery.EXTENSION_SUFFIXES[0]}").write_bytes(b"planted, not a library")
            env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONSAFEPATH")}
            guarded = subprocess.run([sys.executable, str(root / "harness.py")], capture_output=True, text=True,
                                     env=env, cwd=root, check=False)
            self.assertEqual((guarded.returncode, guarded.stdout.strip()), (0, "source"), guarded.stderr)
            plain = subprocess.run(
                [sys.executable, "-B", "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                 "from oracle_harness import verdict; print(verdict.say())", str(root)],
                capture_output=True, text=True, env=env, check=False)
            self.assertNotEqual(plain.returncode, 0, plain.stdout)
            self.assertIn("ImportError", plain.stderr)


if __name__ == "__main__":
    unittest.main()
