"""Integrity checks for the package's declared public API.

These catch the failure mode where a symbol is deleted from its defining
module but left behind in an ``__all__`` list (or vice versa). Nothing here
exercises behaviour -- it only asserts that what the package advertises as
public actually exists.
"""

import importlib

import pytest

import mohilo

SUBPACKAGES = [
    "mohilo.feedback",
    "mohilo.optimization",
    "mohilo.utils",
    "mohilo.performance",
]


def _all_of(module_name):
    module = importlib.import_module(module_name)
    return module, getattr(module, "__all__", None)


@pytest.mark.parametrize("name", mohilo.__all__)
def test_toplevel_all_entries_resolve(name):
    """Every name in ``mohilo.__all__`` is actually importable from mohilo."""
    assert hasattr(mohilo, name), (
        f"'{name}' is listed in mohilo.__all__ but does not exist on the "
        f"package. It was probably deleted from its defining module without "
        f"being removed from __all__."
    )


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_subpackage_all_entries_resolve(module_name):
    """Every name in each subpackage's ``__all__`` resolves."""
    module, declared = _all_of(module_name)
    assert declared is not None, f"{module_name} does not define __all__"

    missing = [name for name in declared if not hasattr(module, name)]
    assert not missing, (
        f"{module_name}.__all__ lists names that do not exist: {missing}"
    )


@pytest.mark.parametrize("module_name", ["mohilo", *SUBPACKAGES])
def test_all_has_no_duplicates(module_name):
    """A name appearing twice in ``__all__`` usually means a bad merge."""
    module = importlib.import_module(module_name)
    declared = list(getattr(module, "__all__", []))
    duplicates = sorted({n for n in declared if declared.count(n) > 1})
    assert not duplicates, f"{module_name}.__all__ has duplicates: {duplicates}"


def test_star_import_surface_matches_all():
    """``from mohilo import *`` yields exactly what ``__all__`` promises."""
    namespace = {}
    exec("from mohilo import *", namespace)  # noqa: S102
    namespace.pop("__builtins__", None)
    assert set(namespace) == set(mohilo.__all__)
