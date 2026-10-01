import os
import subprocess
import unittest

from oracle_harness import lockfile
from oracle_harness.lockfile import normalise_url
from oracle_harness.util import sha256_bytes
from support import FAKE_LICENCE, GIT_ENV, Sandbox, git


def codes(findings):
    return sorted({finding.code for finding in findings})


class UrlTests(unittest.TestCase):
    def test_spellings_of_one_repository_compare_equal(self):
        expected = "github.com/ftxi/5dchess_engine"
        for url in (
            "https://github.com/ftxi/5dchess_engine",
            "https://github.com/ftxi/5dchess_engine.git",
            "git@github.com:ftxi/5dchess_engine.git",
            "ssh://git@github.com/ftxi/5dchess_engine",
            "HTTPS://GitHub.com/ftxi/5dchess_engine/",
        ):
            self.assertEqual(normalise_url(url), expected, url)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox(self)

    def verify(self, **overrides):
        return lockfile.verify(self.box.lock(**overrides), self.box.repo)

    def test_a_faithful_pin_passes(self):
        self.assertEqual(self.verify(), [])

    def test_lock_and_gitlink_must_name_the_same_commit(self):
        self.assertIn("gitlink-mismatch", codes(self.verify(commit="0" * 40)))

    def test_checkout_must_be_at_the_locked_commit(self):
        self.box.write(self.box.upstream / "extra.txt", "newer\n")
        git(self.box.upstream, "add", "-A")
        git(self.box.upstream, "commit", "-q", "-m", "newer")
        git(self.box.checkout, "pull", "-q", "origin", "main")
        self.assertIn("checkout-mismatch", codes(self.verify()))

    def test_third_party_is_read_only(self):
        self.box.write(self.box.checkout / "src" / "rules.cpp", "// patched\n")
        self.assertIn("dirty-submodule", codes(self.verify()))

    def test_untracked_files_in_a_submodule_count_as_changes(self):
        self.box.write(self.box.checkout / "node_modules" / "x.js", "x\n")
        self.assertIn("dirty-submodule", codes(self.verify()))

    def test_a_crlf_checkout_is_not_mistaken_for_local_changes(self):
        # Under core.autocrlf=true (the Windows default) the work tree holds CRLF
        # files for LF blobs. That is a faithful checkout, not a modification -
        # even once git has to re-read a file because its timestamp moved.
        source = self.box.checkout / "src" / "rules.cpp"
        git(self.box.checkout, "config", "core.autocrlf", "true")
        source.unlink()
        done = subprocess.run(["git", "-C", str(self.box.checkout), "checkout", "--", "src/rules.cpp"],
                              capture_output=True, check=True, env={**os.environ, **GIT_ENV})
        self.assertEqual(done.returncode, 0)
        self.assertIn(b"\r\n", source.read_bytes())
        source.write_bytes(source.read_bytes())
        self.assertEqual(self.verify(), [])

    def test_a_changed_licence_is_noticed(self):
        findings = self.verify(licence_sha256="f" * 64)
        self.assertEqual(codes(findings), ["licence-changed"])
        # The new hash in full: after a pin bump it is what the lock file needs.
        self.assertIn(sha256_bytes(FAKE_LICENCE.encode()), findings[0].message)

    def test_a_missing_licence_file_is_noticed(self):
        self.assertIn("licence-missing", codes(self.verify(licence_file="COPYING")))

    def test_the_tree_id_is_checked(self):
        findings = self.verify(tree="1" * 40)
        self.assertEqual(codes(findings), ["tree-mismatch"])
        self.assertIn(git(self.box.upstream, "rev-parse", "HEAD^{tree}"), findings[0].message)

    def test_the_engine_may_have_submodules_the_lock_file_does_not_know(self):
        other = self.box.root / "dependency"
        other.mkdir()
        git(other, "init", "-q", "-b", "main")
        self.box.write(other / "README", "not an oracle\n")
        git(other, "add", "-A")
        git(other, "commit", "-q", "-m", "a dependency")
        git(self.box.repo, "submodule", "add", "-q", "--name", "dependency", other.as_posix(), "engine/extern/dependency")
        self.assertEqual(self.verify(), [])
        # An oracle, though, is recognised wherever it is checked out.
        git(self.box.repo, "submodule", "add", "-q", "--name", "again", self.box.upstream.as_posix(), "engine/extern/again")
        self.assertEqual(codes(self.verify()), ["unlocked-submodule"])

    def test_url_and_path_must_match_gitmodules(self):
        self.assertIn("submodule-url", codes(self.verify(url="https://example.com/other/repo")))
        self.assertIn("submodule-path", codes(self.verify(path="oracles/third_party/elsewhere")))

    def test_every_submodule_must_be_locked(self):
        lock = lockfile.Lock(pins={"fake": self.box.pin(submodule="some-other-name")})
        found = codes(lockfile.verify(lock, self.box.repo))
        self.assertIn("unlocked-submodule", found)
        self.assertIn("missing-submodule", found)

    def test_submodules_live_under_third_party(self):
        git(self.box.repo, "submodule", "add", "-q", "--name", "stray", self.box.upstream.as_posix(), "vendor/stray")
        lock = lockfile.Lock(pins={
            "fake": self.box.pin(),
            "stray": self.box.pin(id="stray", submodule="stray", path="vendor/stray"),
        })
        self.assertIn("submodule-placement", codes(lockfile.verify(lock, self.box.repo)))


class LoadTests(unittest.TestCase):
    def test_the_repository_lock_file_loads(self):
        lock = lockfile.load()
        self.assertEqual({pin.id for pin in lock.oracles()}, {"ftxi", "cwmtt", "chessin5d", "notation"})
        self.assertEqual({lock.pin(name).tier for name in ("ftxi", "notation")}, {"commit"})
        self.assertEqual({lock.pin(name).tier for name in ("cwmtt", "chessin5d")}, {"nightly"})
        # The notation repository's parser reads the notation and judges nothing (H26).
        self.assertEqual([pin.id for pin in lock.oracles() if pin.parse_only], ["notation"])
        self.assertEqual(lock.pin("notation").role, "parse-oracle")


if __name__ == "__main__":
    unittest.main()
