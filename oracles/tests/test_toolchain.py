"""The pinned Rust toolchain (Kairos, decision H52): its pins, its unpacking, and the build environment."""

import io
import os
import re
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle_harness import build, lockfile, paths, toolchain
from oracle_harness.util import HarnessError


def _dist_archive(folder: Path, top: str, components: dict[str, dict[str, bytes]], listed: list[str] | None = None) -> Path:
    """A tar.xz in the shape of a Rust dist archive: one top directory, a components file, a tree per component."""
    archive = folder / f"{top}.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add("install.sh", b"#!/bin/sh\n")
        add("components", ("\n".join(listed if listed is not None else components) + "\n").encode())
        for component, files in components.items():
            add(f"{component}/manifest.in", b"file:bin/x\n")
            for name, data in files.items():
                add(f"{component}/{name}", data)
    return archive


class RustPinTests(unittest.TestCase):
    def setUp(self):
        self.rust = lockfile.load().toolchains["rust"]

    def test_an_exact_release_is_pinned(self):
        self.assertRegex(self.rust["version"], r"^\d+\.\d+\.\d+$")

    def test_every_archive_is_the_pinned_release_with_a_checksum(self):
        self.assertEqual(set(self.rust["dist"]), {"windows-x86_64", "linux-x86_64"})
        for platform, dist in self.rust["dist"].items():
            names = set()
            for component in dist["components"]:
                url = component["url"]
                self.assertTrue(url.startswith("https://static.rust-lang.org/dist/"), url)
                self.assertIn(f"-{self.rust['version']}-{dist['target']}.tar.xz", url, url)
                self.assertRegex(component["sha256"], r"^[0-9a-f]{64}$")
                names.add(re.sub(r"-\d.*$", "", url.rsplit("/", 1)[-1]))
            self.assertTrue({"rustc", "cargo", "rust-std"} <= names, platform)
        # The GNU target on Windows links with the MinGW linker the toolchain ships.
        windows = self.rust["dist"]["windows-x86_64"]
        self.assertEqual(windows["target"], "x86_64-pc-windows-gnu")
        self.assertTrue(any("/rust-mingw-" in component["url"] for component in windows["components"]))


class MergeComponentTests(unittest.TestCase):
    def test_each_component_tree_lands_in_the_prefix_without_its_installer_or_documentation(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            archive = _dist_archive(folder, "rustc-9.9.9-x", {
                "rustc": {"bin/rustc.exe": b"compiler", "lib/librustc.so": b"lib", "share/doc/rustc/README.md": b"doc",
                          "share/man/man1/rustc.1": b"man", "share/zsh/site-functions/_rustc": b"completion"},
                "rust-docs": {"share/doc/rust/html/index.html": b"docs"},
            }, listed=["rustc"])
            prefix = folder / "prefix"
            prefix.mkdir()
            self.assertEqual(toolchain.merge_rust_component(archive, prefix), ["rustc"])
            found = sorted(path.relative_to(prefix).as_posix() for path in prefix.rglob("*") if path.is_file())
            # Only the listed component; no manifest.in, install.sh, documentation or manual pages.
            self.assertEqual(found, ["bin/rustc.exe", "lib/librustc.so", "share/zsh/site-functions/_rustc"])
            self.assertEqual((prefix / "bin" / "rustc.exe").read_bytes(), b"compiler")

    def test_components_merge_into_one_prefix(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            prefix = folder / "prefix"
            prefix.mkdir()
            toolchain.merge_rust_component(_dist_archive(folder, "rustc-x", {"rustc": {"bin/rustc": b"r"}}), prefix)
            toolchain.merge_rust_component(_dist_archive(folder, "cargo-x", {"cargo": {"bin/cargo": b"c"}}), prefix)
            self.assertEqual(sorted(path.name for path in (prefix / "bin").iterdir()), ["cargo", "rustc"])

    def test_an_archive_that_is_not_a_rust_dist_archive_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            archive = folder / "other.tar.xz"
            with tarfile.open(archive, "w:xz") as tar:
                info = tarfile.TarInfo("other/bin/tool")
                info.size = 1
                tar.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(HarnessError):
                toolchain.merge_rust_component(archive, folder)
            with self.assertRaises(HarnessError):
                toolchain.merge_rust_component(_dist_archive(folder, "empty", {}, listed=[]), folder)


class RustEnvironmentTests(unittest.TestCase):
    def test_the_machines_rust_settings_stay_out_of_the_build(self):
        cargo = toolchain.Tool("cargo", Path("/toolchain/bin") / "cargo", "1.99.0")
        rustc = toolchain.Tool("rustc", Path("/toolchain/bin") / "rustc", "1.99.0")
        planted = {"RUSTFLAGS": "-C target-cpu=native", "CARGO_TARGET_DIR": "/elsewhere", "CARGO_BUILD_RUSTFLAGS": "-x",
                   "RUSTC_WRAPPER": "sccache", "RUSTUP_TOOLCHAIN": "nightly", "CARGO_HOME": "/home/me/.cargo"}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(os.environ, {**planted, "ORACLE_HARNESS_CACHE": folder}):
            env = toolchain.rust_env(cargo, rustc, ceiling=Path(folder))
            for name in ("RUSTFLAGS", "CARGO_TARGET_DIR", "CARGO_BUILD_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTUP_TOOLCHAIN"):
                self.assertNotIn(name, env)
            self.assertEqual(Path(env["CARGO_HOME"]), paths.cache_dir() / "toolchains" / "cargo-home")
            self.assertEqual(env["RUSTC"], os.fspath(rustc.path))
            self.assertEqual(env["CARGO_NET_OFFLINE"], "true")
            self.assertTrue(env["PATH"].startswith(os.fspath(cargo.path.parent)))


class KairosRecipeTests(unittest.TestCase):
    def test_the_driver_files_are_the_build_inputs(self):
        inputs = build._driver_inputs("kairos")
        self.assertEqual([path.relative_to(paths.DRIVERS / "kairos").as_posix() for path in inputs],
                         list(build.KAIROS_DRIVER_FILES))
        for path in inputs:
            self.assertTrue(path.exists(), path)

    def test_the_driver_names_the_exported_crate_and_is_its_own_workspace(self):
        manifest = (paths.DRIVERS / "kairos" / "Cargo.toml").read_text(encoding="utf-8")
        self.assertIn('kairos = { path = "../src" }', manifest)
        self.assertRegex(manifest, r"(?m)^\[workspace\]$")
        lock = (paths.DRIVERS / "kairos" / "Cargo.lock").read_text(encoding="utf-8")
        # No dependency beyond the pinned crate: the build runs --locked --offline.
        self.assertEqual(re.findall(r'(?m)^name = "([^"]+)"$', lock), ["kairos", "kairos_oracle_driver"])
        self.assertNotIn("source = ", lock)

    def test_its_recipe_needs_rust_only(self):
        self.assertEqual(build.recipe_tools("kairos"), ("rust",))


if __name__ == "__main__":
    unittest.main()
