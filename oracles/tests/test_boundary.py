import dataclasses
import os
import unittest

from oracle_harness import boundary
from oracle_harness.boundary import comment_family, strip_comments
from oracle_harness.config import Waiver
from support import BOUNDARY, FAKE_ORACLE_SOURCE, FAKE_ORACLE_TABLE, Sandbox, git

URL = "https://example.com/example/fake-oracle.git"


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox(self)
        self.lock = self.box.lock(url=URL)

    def check(self, config=BOUNDARY, **options):
        return boundary.check(self.lock, config, self.box.repo, harness_rel="oracles", **options)

    def codes(self, **options):
        return sorted({finding.code for finding in self.check(**options)})

    def found(self, path: str, text: str | bytes):
        """The rules a single file breaks."""
        if isinstance(text, bytes):
            (self.box.repo / path).parent.mkdir(parents=True, exist_ok=True)
            (self.box.repo / path).write_bytes(text)
            git(self.box.repo, "add", "--", path)
        else:
            self.box.add(path, text)
        return sorted({finding.code for finding in self.check() if finding.message.startswith(path + ":")})

    def each(self, cases: dict[str, str], expected: list[str]):
        for path, text in cases.items():
            self.assertEqual(self.found(path, text), expected, path)

    # --- what is allowed ---------------------------------------------------------

    def test_a_clean_repository_passes(self):
        self.box.add("engine/src/lib.rs", "pub fn perft() -> u64 { 20 }\n")
        self.box.add("engine/Cargo.toml", '[package]\nname = "engine"\n')
        self.box.add("README.md", "See oracles/README.md for the gate.\n")
        self.assertEqual(self.check(), [])

    def test_the_harness_itself_may_do_what_it_likes(self):
        self.box.add("oracles/oracle_harness/x.py", "import oracle_harness\nPATH = 'oracles/third_party/fake'\n")
        self.assertEqual(self.check(), [])

    def test_a_word_that_merely_contains_the_directory_name_is_fine(self):
        self.box.add("engine/src/lib.rs",
                     'const SUB: &str = "suboracles/x";\nlet oracles_v2 = load("my-oracles/data", "oracles.lock");\n')
        self.box.add("engine/check.py", "import oracles_v2\nfrom my_oracles import table\noracle_harness_notes = 1\n")
        self.assertEqual(self.check(), [])

    def test_citing_an_oracle_in_source_or_documents_is_fine(self):
        self.box.add("engine/src/mate.rs", "// Hypercuboid search, after https://example.com/example/fake-oracle (fake-oracle-pkg).\n")
        self.box.add("docs/design.md", "We test against https://example.com/example/fake-oracle in oracles/third_party/fake.\n")
        self.box.add("engine/mate.py", '"""After https://example.com/example/fake-oracle, whose package is fake-oracle-pkg."""\n\n'
                                       "def mate():\n    \"\"\"See oracles/gate.toml and example/fake-oracle.\"\"\"\n")
        self.box.add("CITATION.cff", "references:\n  - url: https://example.com/example/fake-oracle\n")
        self.assertEqual(self.check(), [])

    def test_a_comment_may_point_at_the_harness(self):
        self.box.add("engine/adapter.py", "# Speaks the protocol described in oracles/README.md (oracle_harness).\nimport json\n")
        self.box.add("engine/src/gate.rs",
                     "// see oracles/README.md\n/* and oracles/gate.toml,\n * oracles/golden/ftxi.json */\nfn main() {}\n")
        commented_out = '[dependencies]\n# fake = { git = "https://example.com/example/fake-oracle" }\n# members = ["oracles"]\n'
        self.box.add("engine/Cargo.toml", commented_out)
        self.assertEqual(self.check(), [])

    def test_a_comment_at_the_end_of_a_line_may_point_at_the_harness_too(self):
        self.box.add("engine/adapter.py", "import json  # the observation protocol of oracles/oracle_harness/adapters/base.py\n")
        self.box.add("engine/src/gate.rs", "fn main() {} // mirrors oracles/gate.toml\n")
        self.box.add("engine/src/Rules.hs", "module Rules where -- compared with oracles/third_party/cwmtt\n{- and\n   oracles/golden -}\n")
        self.assertEqual(self.check(), [])

    def test_code_may_cite_the_harness_manual(self):
        self.box.add("engine/cli.py", 'HELP = """Runs the adapter.\n\nThe protocol is described in oracles/README.md."""\n')
        self.box.add("engine/src/lib.rs", 'const DOCS: &str = "see oracles/README.md";\n')
        self.assertEqual(self.check(), [])

    def test_a_recipe_may_run_the_gate(self):
        self.box.add("justfile", "gate:\n    python oracles/harness.py gate --tier nightly --source fake-oracle-pkg --record r/1\n")
        self.box.add("Makefile", "oracles:\n\tpython oracles/harness.py build --oracle fake-oracle-pkg\n"
                                 "check:\n\tpython oracles/harness.py verify\n")
        self.box.add("package.json", '{"scripts": {"gate": "python oracles/harness.py gate --tier commit"}}\n')
        self.box.add(".pre-commit-config.yaml", "repos:\n  - repo: local\n    hooks:\n      - id: gate\n"
                                                "        entry: python oracles/harness.py verify\n")
        self.assertEqual(self.check(), [])

    def test_prose_in_a_manifest_may_use_the_names(self):
        self.box.add("engine/Cargo.toml", '[package]\nname = "engine"\n'
                                          'description = "rules engine, checked against oracles such as fake-oracle-pkg"\n'
                                          'keywords = ["chess", "oracles"]\n')
        self.box.add("package.json", '{"description": "the engine, checked against fake-oracle-pkg", "keywords": ["chess", "oracles"]}\n')
        self.box.add("engine/engine.cabal", "name: engine\ndescription: checked against the oracles\n")
        self.assertEqual(self.check(), [])

    def test_a_list_that_keeps_the_harness_out_may_name_it(self):
        self.box.add("Cargo.toml", '[workspace]\nmembers = ["engine"]\nexclude = ["oracles"]\n')
        self.box.add("pyproject.toml", '[tool.ruff]\nextend-exclude = ["oracles/*", "oracles"]\n')
        self.box.add("pytest.ini", "[pytest]\nnorecursedirs = oracles .git\n")
        self.box.add("tsconfig.json", '{"compilerOptions": {},\n "exclude": [\n   "oracles",\n   "node_modules"\n ]}\n')
        self.box.add("conftest.py", 'collect_ignore = ["oracles"]\n')
        self.box.add("setup.cfg", "[flake8]\nexclude = oracles,build\n")
        self.box.add(".dockerignore", "oracles\n")
        self.box.add(".editorconfig", "[oracles/**]\nindent_size = 4\n")
        self.box.add("CODEOWNERS", "oracles/ @someone\n")
        self.box.add(".vscode/settings.json", '{"files.exclude": {"oracles/.cache": true}}\n')
        self.assertEqual(self.check(), [])

    def test_a_name_that_is_not_the_harness_directory(self):
        # A Cargo feature, a cabal flag, a map key, and the engine's own directory of that name.
        self.box.add("engine/src/lib.rs", '#[cfg(feature = "oracles")]\nmod adapter;\n')
        self.box.add("engine/Cargo.toml", '[package]\nname = "engine"\n[features]\ndefault = ["oracles"]\n')
        self.box.add("engine/engine.cabal", "name: engine\nflag oracles\n  default: False\n")
        self.box.add("engine/config.py", 'SETTINGS = {"oracles": 3}\n')
        self.box.add("engine/src/oracles/tablebase.h", "int probe();\n")
        self.box.add("engine/src/probe.cpp", '#include "oracles/tablebase.h"\n')
        self.box.add("engine/py/oracles/__init__.py", "class TablebaseOracle: ...\n")
        self.box.add("engine/py/main.py", "from oracles import TablebaseOracle\nimport oracles.harness\n")
        self.assertEqual(self.check(), [])
        # The same import with no such package beside it is the harness.
        self.assertEqual(self.found("engine/main.py", "from oracles import TablebaseOracle\n"), ["imports-harness"])

    # --- reaching into the harness -----------------------------------------------

    def test_core_code_may_not_name_a_path_inside_the_harness(self):
        self.box.add("engine/src/lib.rs", 'const X: &str = include_str!("../../oracles/third_party/fake/src/rules.cpp");\n')
        findings = self.check()
        self.assertEqual([finding.code for finding in findings], ["references-harness"])
        self.assertIn("engine/src/lib.rs:1", findings[0].message)

    def test_whatever_the_case_or_the_separator(self):
        self.each({
            "engine/build.py": 'ROOT = r"..\\oracles\\third_party"\n',
            "engine/other.py": 'ROOT = "..\\\\oracles\\\\golden"\n',
            "engine/src/a.rs": 'const X: &str = include_str!("../../Oracles/third_party/fake/src/rules.cpp");\n',
            "engine/src/b.cpp": '#include "../../ORACLES/third_party/fake/src/rules.h"\n',
        }, ["references-harness"])

    def test_but_not_the_oracles_answers_or_their_documents(self):
        # DISAGREEMENTS.md is generated from what the oracles say; a third-party
        # README is the oracle's. Neither is the harness's manual.
        self.each({
            "engine/src/a.rs": 'const T: &str = include_str!("../../oracles/DISAGREEMENTS.md");\n',
            "engine/src/b.rs": 'const T: &str = include_str!("../../oracles/third_party/fake/README.md");\n',
        }, ["references-harness"])

    def test_a_program_may_not_run_the_harness(self):
        # A recipe may run the gate; a program is how the engine would get at an oracle's answers.
        self.each({
            "tools/prepush.py": 'import subprocess\nsubprocess.run(["python", "oracles/harness.py", "verify"])\n',
            "build.rs": 'fn main() { std::process::Command::new("python").arg("oracles/harness.py").arg("verify"); }\n',
        }, ["references-harness"])

    def test_nor_a_recipe_ask_it_for_oracle_answers(self):
        self.assertEqual(self.found("Makefile", "engine/tables.json:\n\tpython oracles/harness.py observe --source ftxi > $@\n"),
                         ["references-harness"])

    def test_nor_change_into_the_harness_to_run_it(self):
        observe = "python harness.py observe --source ftxi --record r/1"
        self.each({
            "tools/tables.py": f'import subprocess\nsubprocess.run("cd oracles && {observe}", shell=True)\n',
            "tools/tables2.py": f'import os\nos.system("pushd ./oracles && {observe}")\n',
            "package.json": f'{{"scripts": {{"tables": "cd oracles && {observe}"}}}}\n',
            "engine/src/build.rs": 'fn main() { run("make -C oracles golden"); run("npm --prefix=oracles ci"); }\n',
        }, ["references-harness"])
        self.box.add("engine/oracles/notes.txt", "the engine's own directory of that name\n")
        self.assertEqual(self.found("engine/gen.py", 'import os\nos.system("cd oracles && make")\n'), [])

    def test_a_build_tool_configuration_is_a_manifest(self):
        self.assertEqual(self.found("web/vite.config.ts", "export default { optimizeDeps: { include: ['fake-oracle-pkg'] } };\n"),
                         ["depends-on-oracle"])

    def test_a_manifest_may_name_nothing_in_the_harness_but_its_entry_point(self):
        self.assertEqual(self.found("Makefile", "data:\n\tcp oracles/golden/ftxi.json engine/tables/\n"), ["references-harness"])

    def test_the_harness_directory_may_not_be_built_into_a_path(self):
        self.each({
            "engine/a.py": "import os, sys\nsys.path.insert(0, os.path.join(ROOT, 'oracles'))\nimport harness\n",
            "engine/b.py": "from pathlib import Path\nTABLES = Path('oracles') / 'golden'\n",
            "engine/src/c.rs": 'let dir = root.join("oracles").join("third_party");\n',
            "engine/src/d.js": "const dir = path.join(root, `oracles`, 'golden');\n",
            "engine/e.py": 'ROOT = "../oracles"\n',
            "engine/f.py": 'TABLES = f"{ROOT}/oracles"\n',
        }, ["references-harness"])

    def test_a_path_that_ends_in_the_harness_directory(self):
        self.each({
            "engine/Makefile": "CXXFLAGS += -I../oracles\n",
            "CMakeLists.txt": "target_include_directories(engine PRIVATE ${CMAKE_SOURCE_DIR}/oracles)\n",
            "build.rs": 'fn main() { let dir = format!("{}/../oracles", env!("CARGO_MANIFEST_DIR")); }\n',
            "pyproject.toml": '[project]\ndependencies = ["harness @ file:///${PROJECT_ROOT}/oracles"]\n',
        }, ["harness-in-build"])

    def test_core_code_may_not_import_the_harness(self):
        self.each({
            "engine/a.py": "from oracle_harness import gate\n",
            "engine/b.py": "import oracles.oracle_harness.gate\n",
            "engine/c.py": "from oracles import harness\n",
            "engine/d.py": "import json, oracles.harness as h\n",
            "engine/e.py": "from oracles.oracle_harness.observations import compare\n",
            "engine/f.py": 'import importlib\ngate = importlib.import_module("oracles.oracle_harness.gate")\n',
            "engine/src/g.rs": "use oracle_harness::gate;\n",
            "engine/h.py": 'import os\nCACHE = os.environ["ORACLE_HARNESS_CACHE"]\n',
            "engine/pyproject.toml": '[project]\ndependencies = ["oracle-harness"]\n',
        }, ["imports-harness"])

    def test_a_hash_that_is_code_is_not_a_comment(self):
        self.box.add("engine/src/a.cpp", '#include "../../oracles/third_party/fake/src/rules.h"\n')
        self.box.add("engine/src/b.rs", '#[path = "../../oracles/third_party/fake/src/rules.rs"]\nmod rules;\n')
        findings = self.check()
        self.assertEqual([(item.code, item.message.split(":")[0]) for item in findings],
                         [("references-harness", "engine/src/a.cpp"), ("references-harness", "engine/src/b.rs")])

    def test_what_is_a_comment_depends_on_the_kind_of_file(self):
        # Each of these lines opens with what is a comment leader in some other language.
        self.each({
            "engine/src/a.c": '*path = "../oracles/third_party/fake/src/rules.cpp";\n',
            "engine/b.sh": "cargo build \\\n  --manifest-path oracles/engine-shim/Cargo.toml\n",
            "engine/c.py": 'x = 1\n-- "oracles/golden/ftxi.json"\n',
            "engine/d.json": '{"tables": "oracles/golden/ftxi.json", "note": "# not a comment in JSON"}\n',
            "engine/e.unknown": "// oracles/golden/ftxi.json\n",
            "engine/engine.cabal": "ld-options: --as-needed -L../oracles/third_party/fake\n",
        }, ["references-harness"])

    def test_instructions_to_tools_are_not_comments(self):
        self.each({
            "engine/src/a.ts": '/// <reference path="../../oracles/third_party/fake/types.d.ts" />\nexport {};\n',
            "engine/b.go": "package b\n\n//go:embed oracles/golden/ftxi.json\nvar tables string\n",
            "engine/C.hs": "{-# OPTIONS_GHC -i../oracles/third_party/fake #-}\nmodule C where\n",
            "engine/d.sh": "cat > x.c <<EOF\n#include \"../oracles/third_party/fake/src/rules.h\"\nEOF\n",
            "engine/paths.h.in": '#include "@PROJECT_SOURCE_DIR@/oracles/third_party/fake/src/rules.h"\n',
        }, ["references-harness"])

    def test_a_string_is_not_a_comment(self):
        self.assertEqual(self.found("engine/a.py", 'URL = "http://x/#oracles/golden/ftxi.json"\nPATH = "# oracles/golden"\n'),
                         ["references-harness"])
        self.assertEqual(self.found("engine/src/b.rs", 'let p = "// oracles/golden/ftxi.json";\n'), ["references-harness"])
        self.assertEqual(self.found("engine/src/c.rs", 'let p = "line one\n// oracles/golden/ftxi.json";\n'), ["references-harness"])

    def test_a_regular_expression_does_not_open_a_comment(self):
        source = "const root = dir.replace(/\\/*$/, '');\nconst rules = require('fake-oracle-pkg');\n"
        self.assertEqual(self.found("engine/src/a.js", source), ["depends-on-oracle"])

    def test_the_line_number_survives_comments_and_line_endings(self):
        self.box.add("engine/check.py", '# one\n"""two\nthree"""\n\nimport oracle_harness  # four\n')
        self.assertIn("engine/check.py:5:", self.check()[0].message)
        self.assertEqual(self.found("engine/old.py", b"# one\r# two\rimport oracle_harness\r"), ["imports-harness"])
        self.assertIn("engine/old.py:3:", next(item for item in self.check() if "old.py" in item.message).message)

    def test_every_offending_line_is_reported(self):
        self.box.add("engine/check.py", "import oracle_harness\nx = 1\nfrom oracle_harness import gate\n")
        self.assertEqual([finding.message.split(": ")[0] for finding in self.check()], ["engine/check.py:1", "engine/check.py:3"])

    # --- the build ---------------------------------------------------------------

    def test_the_harness_may_not_join_the_engine_build(self):
        self.each({
            "Cargo.toml": '[workspace]\nmembers = ["engine", "oracles"]\n',
            "package.json": '{"workspaces": ["engine", "./oracles"]}\n',
            "CMakeLists.txt": "add_subdirectory(engine)\nadd_subdirectory(oracles)\n",
            "Makefile": "SUBDIRS = engine oracles\n",
            "engine/makefile": "SUBDIRS = ../oracles\n",  # the name's case does not matter
            "pnpm-workspace.yaml": "packages:\n  - engine\n  - oracles\n",
            "go.work": "go 1.22\nuse ./engine\nuse ./oracles\n",
            "pytest.ini": "[pytest]\npythonpath = oracles\n",
            "tools/build.sh": "cd oracles && python harness.py observe --source ftxi > ../engine/tables.json\n",
        }, ["harness-in-build"])

    def test_a_manifest_may_not_fetch_an_oracle(self):
        self.box.add("engine/Cargo.toml", '[dependencies]\nfake = { git = "https://example.com/example/fake-oracle" }\n')
        self.assertEqual(self.codes(), ["depends-on-oracle"])

    def test_an_oracle_is_recognised_however_it_is_fetched(self):
        self.each({
            "engine/Cargo.toml": '[dependencies]\nfake = { git = "git@example.com:example/fake-oracle.git" }\n',
            "tools/setup.sh": "#!/bin/sh\ngit clone https://example.com/example/fake-oracle vendor/fake\n",
            "tools/gh.sh": "gh repo clone example/fake-oracle vendor/fake\n",
            "tools/raw.sh": "curl -O https://raw.example.com/example/fake-oracle/main/src/rules.cpp\n",
            "tools/api.py": 'URL = "https://api.example.com/repos/example/fake-oracle/tarball"\n',
            "tools/encoded.sh": "curl https://example.com/api/v4/projects/example%2Ffake-oracle/archive\n",
        }, ["depends-on-oracle"])

    def test_a_manifest_or_script_may_not_name_an_oracle_package(self):
        self.each({
            "web/package.json": '{"dependencies": {"fake-oracle-pkg": "^1.0.0"}}\n',
            "requirements.txt": "fake_oracle_pkg==1.0\n",  # package managers read - and _ as one
            "stack.yaml": "extra-deps:\n  - fake-oracle-pkg-0.1.0.0\n",
            "tools/install.sh": "npm install fake-oracle-pkg\n",
            "Dockerfile.dev": "FROM python:3.12\nRUN pip install fake-oracle-pkg\n",
            "deno.json": '{"imports": {"rules": "npm:fake-oracle-pkg@^1.2.1"}}\n',
            ".github/workflows/build.yml": "jobs:\n  b:\n    steps:\n      - run: npm install fake-oracle-pkg\n",
        }, ["depends-on-oracle"])

    def test_source_may_not_import_an_oracle_package(self):
        self.each({
            "web/src/a.js": "const Rules = require('fake-oracle-pkg');\n",
            "web/src/b.mjs": 'import Rules from "fake-oracle-pkg";\n',
            "web/src/c.js": "const Rules = await import('fake-oracle-pkg');\n",
            "web/src/d.ts": 'import { mate } from "fake-oracle-pkg/src/mate.js";\n',
            "engine/e.py": "import fake-oracle-pkg\n",
            "web/src/f.ts": 'import x from "npm:fake-oracle-pkg@1.2.1";\n',
            "web/src/g.js": 'import x from "https://esm.sh/fake-oracle-pkg";\n',
            "web/src/h.js": "const where = require.resolve('fake-oracle-pkg');\n",
            "web/index.html": '<script src="https://cdn.jsdelivr.net/npm/fake-oracle-pkg/dist/index.js"></script>\n',
            "engine/src/i.cpp": "#include <fake-oracle-pkg/state.h>\n",
            "engine/src/j.rs": '#[link(name = "fake-oracle-pkg")]\nextern "C" {}\n',
            "engine/k.py": '# /// script\n# dependencies = [\n#   "fake-oracle-pkg>=1",\n# ]\n# ///\nimport json\n',
        }, ["depends-on-oracle"])

    def test_a_dotted_module_namespace_counts_as_a_package(self):
        self.lock = self.box.lock(url=URL, dependency_names=("Fake.Oracle.Rules",))
        self.assertEqual(self.found("engine/src/A.hs", "module A where\nimport qualified Fake.Oracle.Rules.Moves as M\n"),
                         ["depends-on-oracle"])
        self.assertEqual(self.found("engine/src/B.hs", "module B where\nimport Fake.Oracle.RulesOfThumb\n"), [])

    def test_plumbing_may_run_the_harness_but_not_fetch_an_oracle(self):
        self.box.add(".github/workflows/ci.yml",
                     "jobs:\n  gate:\n    steps:\n      - run: python oracles/harness.py gate --tier commit\n"
                     "      - uses: actions/upload-artifact@v7\n        with:\n          path: oracles/.cache/reports/\n")
        self.box.add(".github/scripts/report.py", "import sys\nsys.path.insert(0, 'oracles')\nimport oracle_harness\n")
        self.assertEqual(self.check(), [])
        self.assertEqual(self.found(".github/workflows/vendor.yml", "jobs:\n  v:\n    steps:\n"
                                    "      - run: git clone https://example.com/example/fake-oracle engine/vendor\n"),
                         ["depends-on-oracle"])
        self.assertEqual(self.found(".claude/helpers/rules.cjs", "const rules = require('fake-oracle-pkg');\n"), ["depends-on-oracle"])

    # --- third-party code --------------------------------------------------------

    def test_third_party_code_may_not_be_committed_into_the_tree(self):
        self.box.add("oracles/third_party/copied/rules.cpp", "int x;\n")
        self.assertEqual(self.codes(), ["vendored-code"])

    def test_an_oracle_checked_out_as_a_submodule_of_the_engine_is_refused(self):
        self.lock = self.box.lock()  # pinned at the upstream's own URL, so the submodule below is recognised
        git(self.box.repo, "submodule", "add", "-q", "--name", "stray", self.box.upstream.as_posix(), "engine/vendor/fake")
        findings = self.check()
        self.assertEqual([finding.code for finding in findings if finding.message.startswith("engine/vendor/fake")],
                         ["depends-on-oracle"])

    def test_the_engine_may_have_submodules_of_its_own(self):
        other = self.box.root / "pybind11"
        other.mkdir()
        git(other, "init", "-q", "-b", "main")
        self.box.write(other / "README", "not an oracle\n")
        git(other, "add", "-A")
        git(other, "commit", "-q", "-m", "a dependency")
        git(self.box.repo, "submodule", "add", "-q", "--name", "pybind11", other.as_posix(), "engine/extern/pybind11")
        self.assertEqual(self.check(), [])

    def test_an_unlocked_submodule_under_third_party_is_refused(self):
        git(self.box.repo, "submodule", "add", "-q", "--name", "extra", self.box.upstream.as_posix(), "oracles/third_party/extra")
        self.assertIn("stray-submodule", self.codes())

    def test_a_repository_dropped_into_the_tree_is_noticed(self):
        nested = self.box.repo / "engine" / "vendor" / "fake"
        nested.mkdir(parents=True)
        git(nested, "init", "-q", "-b", "main")
        self.box.write(nested / "rules.cpp", "int x;\n")
        self.assertEqual(self.codes(), ["embedded-repository"])

    def test_copied_oracle_code_is_found(self):
        body = "\n".join(FAKE_ORACLE_SOURCE.splitlines()[2:12])
        self.box.add("engine/src/count.cpp", "// our own counter\n#include <vector>\n" + body + "\n")
        findings = self.check()
        self.assertEqual([finding.code for finding in findings], ["copied-oracle-code"])
        self.assertIn("example/fake-oracle:src/rules.cpp", findings[0].message)

    def test_each_copied_run_is_reported_once(self):
        lines = FAKE_ORACLE_SOURCE.splitlines()
        first, second = "\n".join(lines[2:9]), "\n".join(lines[3:13])
        ours = "\n".join(f"int our_own_line_number_{n} = compute_something({n});" for n in range(4))
        self.box.add("engine/src/count.cpp", first + "\n" + ours + "\n" + second + "\n")
        findings = self.check()
        self.assertEqual([finding.code for finding in findings], ["copied-oracle-code"] * 2)
        self.assertEqual([finding.message.split(": ")[0] for finding in findings], ["engine/src/count.cpp:1", "engine/src/count.cpp:12"])

    def test_reformatting_a_copy_does_not_hide_it(self):
        lines = FAKE_ORACLE_SOURCE.splitlines()[2:12]
        reindented = "\n".join("\t\t" + line.strip() for line in lines)
        respaced = "\n".join(line.replace(" (", "(").replace(", ", ",").replace(" = ", "=") for line in lines)
        self.box.add("engine/src/a.cpp", reindented + "\n")
        self.box.add("engine/src/b.cpp", respaced + "\n")
        self.box.add("engine/src/c.cpp", ("\r\n".join(lines) + "\r\n").replace("\r\n", "\r"))  # old Mac line endings
        self.assertEqual({finding.message.split(":")[0] for finding in self.check()},
                         {"engine/src/a.cpp", "engine/src/b.cpp", "engine/src/c.cpp"})

    def test_copies_are_looked_for_everywhere(self):
        body = "\n".join(FAKE_ORACLE_SOURCE.splitlines()[2:12]) + "\n"
        for path in (".github/copied/count.cpp", "engine/src/rules.md", "engine/build.rs"):
            self.assertEqual(self.found(path, body), ["copied-oracle-code"], path)

    def test_a_few_shared_lines_are_not_a_copy(self):
        body = "\n".join(FAKE_ORACLE_SOURCE.splitlines()[3:6])
        self.box.add("engine/src/count.cpp", body + "\n")
        self.assertEqual(self.check(), [])

    def test_a_long_line_copied_whole_is_a_copy_on_its_own(self):
        # A data table is one line; six-line windows never see it.
        table = FAKE_ORACLE_TABLE.replace("CENTRALITY", "WEIGHTS").replace(", ", ",")
        self.assertEqual(self.found("engine/src/eval.cpp", f"#include <array>\n{table}int eval();\n"), [])  # renamed
        self.assertEqual(self.found("engine/src/eval2.cpp", f"#include <array>\n{FAKE_ORACLE_TABLE.replace(', ', ',')}int eval();\n"),
                         ["copied-oracle-code"])
        found = [finding.message for finding in self.check() if finding.message.startswith("engine/src/eval2.cpp")]
        self.assertIn("example/fake-oracle:src/tables.cpp:1", found[0])

    def test_comments_and_includes_never_count_as_copied_logic(self):
        header = "\n".join(f"// licence line number {n} of a long standard header" for n in range(12))
        self.box.write(self.box.upstream / "src" / "header.cpp", header + "\n")
        git(self.box.upstream, "add", "-A")
        git(self.box.upstream, "commit", "-q", "-m", "header")
        git(self.box.checkout, "pull", "-q", "origin", "main")
        self.lock = self.box.lock(url=URL, commit=git(self.box.upstream, "rev-parse", "HEAD"))
        self.box.add("engine/src/lib.cpp", header + "\nint ours();\n")
        self.assertEqual(self.check(), [])

    def test_an_oracle_that_cannot_be_read_is_not_silently_skipped(self):
        # Pinned at a commit the checkout does not have: its code cannot be fingerprinted.
        self.lock = self.box.lock(url=URL, commit="0" * 40)
        self.assertEqual(self.check(), [])  # nothing to compare yet: no file outside the harness
        self.box.add("engine/src/count.cpp", "int ours();\n")
        findings = self.check()
        self.assertEqual([finding.code for finding in findings], ["copy-check-skipped"])
        self.assertIn("fake is not checked out at its locked commit", findings[0].message)

    # --- files and links ---------------------------------------------------------

    def test_untracked_files_are_checked_before_they_are_committed(self):
        self.box.write(self.box.repo / "engine" / "check.py", "import oracle_harness\n")
        self.assertEqual(self.codes(), ["imports-harness"])
        self.assertEqual(self.codes(include_untracked=False), [])

    def test_a_file_that_cannot_be_read_as_text_is_reported(self):
        self.assertEqual(self.found("engine/src/a.cpp", b"int x;\0// " + b"y" * 20 + b"\n"), ["unscanned-core-file"])
        big = b"int x;\n" * (boundary._MAX_BYTES // 7 + 1)
        self.assertEqual(self.found("engine/src/big.cpp", big), ["unscanned-core-file"])
        self.assertEqual(self.found("engine/logo.png", b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"), [])  # an image is not code
        weights = b"\x93NUMPY\x01\0" + b"\0" * boundary._MAX_BYTES
        self.assertEqual(self.found("engine/nets/t1.npy", weights), [])  # nor is a large binary file

    def test_a_script_in_utf16_is_read(self):
        script = "git clone https://example.com/example/fake-oracle vendor\n".encode("utf-16")
        self.assertEqual(self.found("tools/fetch.ps1", script), ["depends-on-oracle"])

    def test_a_link_into_the_harness_is_refused(self):
        target = os.path.join("..", "oracles", "third_party", "fake", "src")
        link = self.box.repo / "engine" / "rules"
        link.parent.mkdir(parents=True)
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create symlinks on this machine")
        self.assertEqual(self.codes(), ["symlink-into-harness"])

    def test_a_link_in_the_index_is_refused_where_links_are_checked_out_as_files(self):
        # With core.symlinks=false (the Windows default) git materialises a link
        # as a small text file holding its target.
        for number, target in enumerate(("../oracles/third_party/fake/src", "../Oracles/golden",
                                         "../../repo/oracles/third_party/fake/src")):  # out of the repository and back in
            path = f"engine/rules{number}"
            self.box.write(self.box.repo / path, target)
            blob = git(self.box.repo, "hash-object", "-w", path)
            git(self.box.repo, "update-index", "--add", "--cacheinfo", f"120000,{blob},{path}")
        self.assertEqual([finding.code for finding in self.check()], ["symlink-into-harness"] * 3)

    # --- waivers -----------------------------------------------------------------

    def waived(self, *waivers):
        return dataclasses.replace(BOUNDARY, waivers=tuple(waivers))

    def test_a_waiver_excuses_the_text_it_names_and_nothing_else(self):
        self.box.add("tools/prepush.py", 'RUN = ["python", "oracles/harness.py", "verify"]\nTABLES = "oracles/golden/ftxi.json"\n'
                                         'BOTH = ["oracles/harness.py", "oracles/golden/cwmtt.json"]  # see oracles/harness.py\n')
        self.box.add("tools/other.py", 'RUN = ["python", "oracles/harness.py", "verify"]\n')
        waiver = Waiver("tools/prepush.py", "references-harness", "oracles/harness.py", "a developer script that runs the gate")
        findings = self.check(config=self.waived(waiver))
        # The same rule on another path of the same line, on another line, and the same text in another file, still fail.
        self.assertEqual([(finding.message.split(": ")[0], finding.message.split(": ")[1].split(",")[0]) for finding in findings],
                         [("tools/other.py:1", "names oracles/harness.py"),
                          ("tools/prepush.py:2", "names oracles/golden/ftxi.json"),
                          ("tools/prepush.py:3", "names oracles/golden/cwmtt.json")])

    def test_a_waiver_is_for_one_rule(self):
        self.box.add("engine/a.py", "import oracle_harness\n")
        waiver = Waiver("engine/a.py", "references-harness", "oracle_harness", "the wrong rule")
        self.assertEqual(sorted(finding.code for finding in self.check(config=self.waived(waiver))), ["imports-harness", "waiver-unused"])

    def test_a_waiver_that_excuses_nothing_fails(self):
        self.box.add("engine/a.py", "x = 1\n")
        waiver = Waiver("engine/a.py", "imports-harness", "import oracle_harness", "was needed once")
        findings = self.check(config=self.waived(waiver))
        self.assertEqual([finding.code for finding in findings], ["waiver-unused"])
        self.assertIn("engine/a.py", findings[0].message)

    def test_a_whole_file_finding_is_waived_by_its_path(self):
        self.box.add("oracles/third_party/copied/rules.cpp", "int x;\n")
        waiver = Waiver("oracles/third_party/copied/rules.cpp", "vendored-code", "copied/rules.cpp", "a test of the mechanism")
        self.assertEqual(self.check(config=self.waived(waiver)), [])


class CommentTests(unittest.TestCase):
    def test_families(self):
        cases = {
            "engine/src/lib.rs": "rust", "a/b.CPP": "c", "web/x.ts": "js", "web/y.mts": "js", "go.mod": "slash",
            "engine/x.py": "python", "Makefile": "hash", "engine/Cargo.toml": "hash", "ci/run.sh": "hash",
            "CMakeLists.txt": "hash", "requirements.txt": "hash", "justfile": "hash", "Cargo.lock": "hash",
            "Dockerfile.dev": "hash", "engine/paths.h.in": "c", "engine/bindings.rs.in": "rust",
            "engine/src/Rules.hs": "haskell", "engine/src/Mate.lhs": "literate-haskell", "engine.cabal": "cabal",
            "xmake.lua": "lua", "schema.sql": "sql", "build.bat": "batch", "engine.csproj": "markup",
            "package.json": "none", "notes.unknown": "none", "LICENSE": "none",
        }
        for path, family in cases.items():
            self.assertEqual(comment_family(path), family, path)

    def test_comments_are_blanked_and_everything_keeps_its_place(self):
        text = 'a = 1  # one\n"""doc\n# not a comment"""\nb = "#x" # two\n'
        stripped = strip_comments(text, "hash")
        self.assertEqual(stripped, 'a = 1       \n"""doc\n# not a comment"""\nb = "#x"      \n')
        self.assertEqual(len(stripped), len(text))

    def test_slash_comments(self):
        text = 'x = "//a"; // b\n/* c\n   d */ y = \'/*\'; z = `\n// e`;\n'
        self.assertEqual(strip_comments(text, "js"), 'x = "//a";     \n    \n        y = \'/*\'; z = `\n// e`;\n')

    def test_a_block_comment_that_never_closes_is_not_a_comment(self):
        self.assertEqual(strip_comments("x = a /* b\ny = c\n", "c"), "x = a /* b\ny = c\n")

    def test_haskell_comments(self):
        text = 'x = "--a" -- b\n{- c\n d -} y = a --> b\n{-# LANGUAGE Strict #-}\n'
        self.assertEqual(strip_comments(text, "haskell"), 'x = "--a"     \n    \n      y = a --> b\n{-# LANGUAGE Strict #-}\n')

    def test_literate_haskell_is_code_only_where_marked(self):
        text = "Prose about oracles/x\n> import Data.List -- a comment\n> main = oracles\n"
        self.assertEqual(strip_comments(text, "literate-haskell"), " " * 21 + "\n  import Data.List             \n  main = oracles\n")

    def test_a_hash_inside_a_word_or_before_a_directive_is_code(self):
        text = 'echo $# ${#name} a#b\nurl = https://x/y#frag # real\n#include "x.h"\n'
        self.assertEqual(strip_comments(text, "hash"), 'echo $# ${#name} a#b\nurl = https://x/y#frag       \n#include "x.h"\n')

    def test_a_lone_quote_does_not_swallow_the_rest(self):
        self.assertEqual(strip_comments("fn f<'a>(x: &'a str) {} // c\n", "rust"), "fn f<'a>(x: &'a str) {}     \n")
        self.assertEqual(strip_comments("msg: don't # c\n", "hash"), "msg: don't    \n")

    def test_a_python_docstring_is_prose_but_a_string_value_is_not(self):
        text = '"""Module."""\n\ndef f():\n    """Doc."""\n    return """value"""\n\nX = (\n    """argument"""\n)\n'
        stripped = strip_comments(text, "python")
        self.assertNotIn("Module", stripped)
        self.assertNotIn("Doc", stripped)
        self.assertIn('"""value"""', stripped)
        self.assertIn('"""argument"""', stripped)

    def test_other_families(self):
        self.assertEqual(strip_comments("name: x\n-- a comment\nld-options: --as-needed -lfoo\n", "cabal"),
                         "name: x\n" + " " * 12 + "\nld-options: --as-needed -lfoo\n")
        self.assertEqual(strip_comments("local t = {-1, 0, 1} -- c\nrequire('x')\n--[[ long\ncomment ]]\n", "lua"),
                         "local t = {-1, 0, 1}     \nrequire('x')\n" + " " * 9 + "\n" + " " * 10 + "\n")
        self.assertEqual(strip_comments("REM oracles/x\n:: oracles/y\necho oracles/z\n", "batch"),
                         " " * 13 + "\n" + " " * 12 + "\necho oracles/z\n")
        self.assertEqual(strip_comments("<a/><!-- oracles/x --><b/>\n", "markup"), "<a/>" + " " * 18 + "<b/>\n")

    def test_a_file_of_unknown_kind_has_no_comments(self):
        text = "// a\n# b\n-- c\n"
        self.assertEqual(strip_comments(text, "none"), text)


class ClassifyTests(unittest.TestCase):
    def test_kinds(self):
        cases = {
            "oracles/harness.py": "harness",
            "oracles": "harness",
            "oracles_notes.txt": "source",
            "README.md": "documentation",
            "docs/deep/notes.md": "documentation",
            ".gitmodules": "plumbing",
            ".github/workflows/ci.yml": "plumbing",
            "engine/Cargo.toml": "manifest",
            "CMakeLists.txt": "manifest",
            "engine/src/lib.rs": "source",
        }
        for path, expected in cases.items():
            self.assertEqual(boundary.classify(path, BOUNDARY, "oracles"), expected, path)


class RepositoryBoundaryConfigTests(unittest.TestCase):
    def test_this_repository_classifies_its_own_files_sensibly(self):
        from oracle_harness import config

        policy = config.load().boundary
        expected = {
            "docs/design/5d-chess-decisions.md": "documentation",
            "docs/design/tools/gen.py": "source",
            "AGENTS.md": "documentation",
            "README.txt": "documentation",
            "CITATION.cff": "documentation",
            ".github/workflows/oracle-commit.yml": "plumbing",
            ".claude/settings.json": "plumbing",
            ".dockerignore": "plumbing",
            ".pre-commit-config.yaml": "plumbing",
            "oracles/gate.toml": "harness",
            "engine/Cargo.toml": "manifest",
            "pyproject.toml": "manifest",
            "justfile": "manifest",
            "GNUmakefile": "manifest",
            "makefile": "manifest",
            "Dockerfile.dev": "manifest",
            "tools/fetch.ps1": "manifest",
            "engine/src/movegen.rs": "source",
        }
        for path, kind in expected.items():
            self.assertEqual(boundary.classify(path, policy, "oracles"), kind, path)


if __name__ == "__main__":
    unittest.main()
