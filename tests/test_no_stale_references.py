"""Static staleness checks for code outside the package.

``examples/``, ``scripts/`` and ``hilo/`` are executable scripts -- importing
them would run full experiments and open plot windows -- so this module parses
them with :mod:`ast` instead and never executes anything.

Two kinds of reference are checked:

1. import statements  -- ``from pypolar.feedback import IdealPoint``
2. alias attributes   -- ``import pypolar as plr`` ... ``plr.BasicGP``

Both go stale the same way when the package is refactored: a module is renamed
(``pypolar.oracles`` -> ``pypolar.feedback``) or a symbol is removed.
"""

# TODO: have this target the correct files
import ast
import importlib
import pathlib

import pytest

import pypolar

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SEARCH_DIRS = ["examples", "scripts", "hilo"]
PACKAGE = "pypolar"


def _python_files():
    for directory in SEARCH_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def _module_exists(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def _is_package_module(name):
    return name == PACKAGE or name.startswith(PACKAGE + ".")


def _collect_problems(path):
    """Return a list of human-readable problems for one file."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"line {exc.lineno}: file does not parse ({exc.msg})"]

    problems = []
    aliases = set()

    for node in ast.walk(tree):
        # import pypolar as plr  /  import pypolar.feedback
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_package_module(alias.name):
                    continue
                if not _module_exists(alias.name):
                    problems.append(
                        f"line {node.lineno}: no module named '{alias.name}'"
                    )
                    continue
                if alias.name == PACKAGE:
                    aliases.add(alias.asname or alias.name)

        # from pypolar.feedback import IdealPoint
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module is None:
                continue
            if not _is_package_module(node.module):
                continue
            if not _module_exists(node.module):
                problems.append(
                    f"line {node.lineno}: no module named '{node.module}'"
                )
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                if not hasattr(module, alias.name):
                    problems.append(
                        f"line {node.lineno}: '{node.module}' has no "
                        f"attribute '{alias.name}'"
                    )

    # plr.BasicGP -- only the first attribute after a known package alias
    if aliases:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.value, ast.Name):
                continue
            if node.value.id not in aliases:
                continue
            if not hasattr(pypolar, node.attr):
                problems.append(
                    f"line {node.lineno}: pypolar has no attribute "
                    f"'{node.attr}' (used as '{node.value.id}.{node.attr}')"
                )

    return problems


ALL_FILES = list(_python_files())


@pytest.mark.skipif(not ALL_FILES, reason="no example/script files found")
@pytest.mark.parametrize(
    "path", ALL_FILES, ids=[str(p.relative_to(REPO_ROOT)) for p in ALL_FILES]
)
def test_no_stale_pypolar_references(path):
    """Every pypolar module/symbol referenced by a script still exists."""
    problems = _collect_problems(path)
    assert not problems, "{}:\n  {}".format(
        path.relative_to(REPO_ROOT), "\n  ".join(problems)
    )
