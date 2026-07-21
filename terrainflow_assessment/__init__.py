def classFactory(iface):
    from .qgis.plugin import TerrainFlowAssessmentPlugin
    return TerrainFlowAssessmentPlugin(iface)
