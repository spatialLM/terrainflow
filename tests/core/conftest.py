"""
Conftest for tests/core/ — enforces that no QGIS modules are imported.
These tests must run on plain pytest with no QGIS runtime.
"""
import sys

import pytest


@pytest.fixture(autouse=True)
def assert_no_qgis_import():
    qgis_before = {k for k in sys.modules if k.startswith("qgis")}
    yield
    qgis_after = {k for k in sys.modules if k.startswith("qgis")}
    new_qgis = qgis_after - qgis_before
    assert not new_qgis, f"Core test imported qgis modules: {new_qgis}"
