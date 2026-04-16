"""
Tests for FlowAnalysis.__init__ — covers the class definition stmts that don't
need pysheds runtime (load_dem/run are blocked by pysheds NumPy 2.0 issue).
"""
import pytest


class TestPluginFlowAnalysisInit:
    def test_init_grid_none(self):
        from plugin.processing.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        assert fa.grid is None

    def test_init_dem_none(self):
        from plugin.processing.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        assert fa.dem is None

    def test_init_fdir_none(self):
        from plugin.processing.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        assert fa.fdir is None

    def test_init_acc_none(self):
        from plugin.processing.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        assert fa.acc is None

    def test_init_routing(self):
        from plugin.processing.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        assert fa.routing == "dinf"

    def test_run_raises_without_dem(self):
        from plugin.processing.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        with pytest.raises(RuntimeError, match="DEM not loaded"):
            fa.run()


class TestAssessmentFlowAnalysisInit:
    def test_init_grid_none(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        assert fa.grid is None

    def test_init_routing(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        assert fa.routing == "dinf"

    def test_run_raises_without_dem(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        with pytest.raises(RuntimeError, match="DEM not loaded"):
            fa.run()
