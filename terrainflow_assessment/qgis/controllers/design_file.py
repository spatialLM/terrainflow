"""
design_file.py — DesignFileController: save and reopen a whole design as one file.

A design previously survived only inside a QGIS project, and only partially: the
earthworks round-tripped but nothing carried the inputs that sized them, and every
reference was an absolute path — so a project opened on another machine resolved no DEM,
no boundary, and no storm.

This controller owns the portable `.tfd` archive that fixes that. The schema itself lives
in :mod:`terrainflow_assessment.modules.project_io` (pure, tested); everything here is
the QGIS half — file dialogs, layer round-tripping, and resolving the DEM.

Cross-controller work is emitted as signals rather than called directly, because the
controllers deliberately do not hold references to each other (see ``_state.py``): the
plugin wires ``earthworks_payload_ready`` and friends to the controllers that own them.
"""

from __future__ import annotations

import os
import zipfile

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsGeometry,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QObject, pyqtSignal
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox

from terrainflow_assessment.modules.dem_loader import (
    clip_dem_to_polygon,
    fingerprint_dem,
    load_dem,
)
from terrainflow_assessment.modules.project_io import (
    AREA_KEYS,
    DemReference,
    DesignDocument,
)

# Archive member names. Fixed rather than derived so a human can unzip a design and know
# what they are looking at.
_MANIFEST_MEMBER = "design.json"
_DEM_MEMBER = "dem.tif"

FILE_FILTER = "TerrainFlow design (*.tfd)"

# How far beyond the site boundary an embedded clip reaches. Without a margin the clipped
# DEM would end exactly at the boundary, and baseline would lose the downslope ground that
# exit points and overflow routing depend on — the design would silently re-open with
# water leaving nowhere.
_CLIP_BUFFER_M = 100.0


class DesignFileError(RuntimeError):
    """A design file could not be written, read, or resolved against a DEM."""


class DesignFileController(QObject):
    """Saves the session to a `.tfd` archive and restores it."""

    # Restored payloads handed to the controllers that own them.
    earthworks_payload_ready = pyqtSignal(str)
    idf_payload_ready = pyqtSignal(str)
    # Emitted when the user accepts the post-restore offer to re-run baseline.
    baseline_rerun_requested = pyqtSignal()

    def __init__(self, state, panel, project, iface, canvas):
        super().__init__()
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas
        # Guards against a second Open landing halfway through the first.
        self._restoring = False

    # ------------------------------------------------------------------ Save

    def save_design(self):
        """Prompt for a path and an embed choice, then write the archive."""
        if not self._state.dem_path:
            self._warn("Load a DEM before saving a design.")
            return

        suggested = f"{self._panel.site_name or 'design'}.tfd".replace(" ", "_")
        path, _filter = QFileDialog.getSaveFileName(
            self._panel, "Save TerrainFlow design", suggested, FILE_FILTER)
        if not path:
            return
        if not path.lower().endswith(".tfd"):
            path += ".tfd"

        embed = self._ask_embed_dem()
        if embed is None:
            return

        try:
            count = self._write_design(path, embed_dem=embed)
        except Exception as exc:
            self._critical(f"Could not save the design: {exc}")
            print(f"TerrainFlow Assessment — design save failed: {exc}")
            return

        where = "with the DEM embedded" if embed else "referencing the DEM by fingerprint"
        self._info(f"Saved {count} earthwork{'s' if count != 1 else ''} {where}.")

    def _ask_embed_dem(self):
        """Ask whether to embed the DEM. Returns True, False, or None if cancelled.

        Asked here rather than set as a panel option because the answer depends on what
        the file is *for* — moving machines or committing a fixture wants the DEM inside;
        saving beside the data it came from does not.
        """
        box = QMessageBox(self._panel)
        box.setWindowTitle("Save design")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("Embed the DEM in the design file?")
        box.setInformativeText(
            "Embedded: a clip of the DEM travels inside the file, so it opens on any "
            "machine with no prompting. Larger file, and analysis re-runs on the clipped "
            "extent.\n\n"
            "Referenced: only a fingerprint is stored. Small file, but the DEM must be "
            "present when the design is opened."
        )
        embed_btn = box.addButton("Embed DEM", QMessageBox.ButtonRole.AcceptRole)
        ref_btn = box.addButton("Reference only", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(embed_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked is embed_btn:
            return True
        if clicked is ref_btn:
            return False
        return None

    def _write_design(self, path, embed_dem):
        """Build the document and write the archive. Returns the earthwork count."""
        dem_path = self._state.dem_path
        reference = fingerprint_dem(dem_path, info=self._state.dem_info)

        embedded_raster = None
        if embed_dem:
            embedded_raster, reference = self._prepare_embedded_dem(dem_path, reference)

        manager = self._state.earthwork_manager
        document = DesignDocument.build(
            inputs=self._panel.collect_inputs(),
            areas=self._collect_areas(),
            dem=DemReference.from_dict(reference),
            earthworks_json=manager.to_json() if manager is not None else None,
            idf=self._idf_json(),
        )

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(_MANIFEST_MEMBER, document.to_json())
            if embedded_raster:
                archive.write(embedded_raster, _DEM_MEMBER)

        return document.earthwork_count()

    def _prepare_embedded_dem(self, dem_path, reference):
        """Clip the DEM for embedding and re-fingerprint the clip.

        The clip is a different raster, so it gets its own fingerprint — the original's
        would claim an identity the embedded pixels do not have.
        """
        reference = dict(reference)
        reference["mode"] = "embedded"
        reference["filename"] = _DEM_MEMBER

        boundary = self._area_geometry("boundary") or self._area_geometry("analysis")
        if boundary is None:
            # Nothing to clip against: embed the DEM whole rather than refuse the save.
            reference["is_clip"] = False
            return dem_path, reference

        try:
            from shapely import wkt as shapely_wkt

            polygon = shapely_wkt.loads(boundary["wkt"]).buffer(_CLIP_BUFFER_M)
            clipped = os.path.join(self._state.output_dir, "design_dem_clip.tif")
            clip_dem_to_polygon(dem_path, polygon, clipped)
        except Exception as exc:
            print(f"TerrainFlow Assessment — DEM clip failed, embedding whole: {exc}")
            reference["is_clip"] = False
            return dem_path, reference

        clipped_reference = fingerprint_dem(clipped)
        clipped_reference.update({
            "mode": "embedded",
            "filename": _DEM_MEMBER,
            "is_clip": True,
            "clip_buffer_m": _CLIP_BUFFER_M,
            # Keep pointing at the DEM this came from, for the "locate" message.
            "original_path": dem_path,
        })
        return clipped, clipped_reference

    def _idf_json(self):
        table = self._state.idf_table
        try:
            return table.to_json() if table is not None else None
        except Exception:
            return None

    # ------------------------------------------------------------------ Open

    def open_design(self):
        """Prompt for a design file, restore it, then offer to re-run baseline."""
        if self._restoring:
            self._warn("A design is still being opened.")
            return

        path, _filter = QFileDialog.getOpenFileName(
            self._panel, "Open TerrainFlow design", "", FILE_FILTER)
        if not path:
            return

        self._restoring = True
        try:
            document = self._restore_design(path)
        except DesignFileError as exc:
            self._critical(str(exc))
            return
        except Exception as exc:
            self._critical(f"Could not open the design: {exc}")
            print(f"TerrainFlow Assessment — design open failed: {exc}")
            return
        finally:
            self._restoring = False

        if document.is_from_newer_build():
            self._warn(
                "This design was saved by a newer version of TerrainFlow. It has been "
                "opened, but any settings this version does not recognise were skipped."
            )

        count = document.earthwork_count()
        self._info(
            f"Restored {count} earthwork{'s' if count != 1 else ''}. "
            "Baseline has not run yet, so the design is not scored."
        )
        self._offer_baseline_rerun()

    def _restore_design(self, path):
        """Read *path* and push it into the session. Returns the document."""
        if not zipfile.is_zipfile(path):
            raise DesignFileError(
                "That file is not a TerrainFlow design (expected a .tfd archive).")

        with zipfile.ZipFile(path) as archive:
            try:
                manifest = archive.read(_MANIFEST_MEMBER).decode("utf-8")
            except KeyError:
                raise DesignFileError(
                    "This design file is missing its manifest and cannot be opened."
                ) from None

            document = DesignDocument.from_json(manifest)
            if document is None:
                raise DesignFileError(
                    "This design file's manifest is corrupt and cannot be read.")

            # DEM first: everything else is meaningless against the wrong terrain, and a
            # failure here must abort before the session is half-overwritten.
            dem_path = self._resolve_dem(document.dem, archive)

        self._apply_dem(dem_path)
        self._panel.apply_inputs(document.inputs)
        self._restore_areas(document.areas)

        if document.idf:
            self.idf_payload_ready.emit(document.idf)
        self.earthworks_payload_ready.emit(document.earthworks_json())
        return document

    # ------------------------------------------------------------------ DEM resolution

    def _resolve_dem(self, reference, archive):
        """Return a usable DEM path for *reference*, or raise :class:`DesignFileError`.

        The single place the two carriage modes converge: nothing downstream branches on
        ``mode``. Extraction and location both end in the same verification, so an
        embedded clip and a hand-located file are held to one standard.
        """
        if reference.mode == "embedded":
            return self._extract_embedded_dem(reference, archive)

        found = self._find_loaded_dem(reference)
        if found:
            return found
        return self._ask_user_to_locate_dem(reference)

    def _extract_embedded_dem(self, reference, archive):
        member = reference.filename or _DEM_MEMBER
        try:
            payload = archive.read(member)
        except KeyError:
            raise DesignFileError(
                "This design says it contains a DEM, but the file is missing from the "
                "archive. It may have been truncated in transit."
            ) from None

        target = os.path.join(self._state.output_dir, "restored_dem.tif")
        with open(target, "wb") as handle:
            handle.write(payload)
        return target

    def _find_loaded_dem(self, reference):
        """Path of an already-loaded raster whose contents match *reference*, if any."""
        if not reference.fingerprint:
            return None
        for layer in self._project.instance().mapLayers().values():
            source = self._raster_source(layer)
            if not source or not os.path.exists(source):
                continue
            try:
                candidate = DemReference.from_dict(fingerprint_dem(source))
            except Exception:
                continue
            if reference.matches(candidate):
                return source
        return None

    @staticmethod
    def _raster_source(layer):
        """The on-disk path of *layer* if it is a file-backed raster, else None."""
        try:
            # 1 == QgsMapLayer.RasterLayer; compared numerically to avoid importing the
            # enum, whose location has moved between QGIS releases.
            if int(layer.type()) != 1:
                return None
            source = layer.source()
        except Exception:
            return None
        # Raster sources can carry provider suffixes (e.g. "path|band=1").
        return source.split("|", 1)[0] if source else None

    def _ask_user_to_locate_dem(self, reference):
        """Ask for the DEM, then verify it is the one the design was built on."""
        hint = os.path.basename(reference.original_path or "") or "the DEM"
        QMessageBox.information(
            self._panel, "Locate the DEM",
            f"This design references {hint}, which is not currently loaded.\n\n"
            "Choose the DEM file to continue. It will be checked against the design.",
        )
        path, _filter = QFileDialog.getOpenFileName(
            self._panel, "Locate the DEM for this design", "",
            "Raster (*.tif *.tiff *.asc *.vrt);;All files (*)")
        if not path:
            raise DesignFileError("Opening cancelled — a design needs its DEM.")

        try:
            candidate = DemReference.from_dict(fingerprint_dem(path))
        except Exception as exc:
            raise DesignFileError(f"That file could not be read as a DEM: {exc}") from exc

        if not reference.matches(candidate):
            raise DesignFileError(
                "That DEM is not the one this design was built on, so the design's "
                "numbers would not apply to it. Opening was stopped rather than "
                "silently re-scoring the design against different terrain."
            )
        return path

    def _apply_dem(self, dem_path):
        """Point the session at *dem_path* and invalidate anything derived from terrain."""
        self._state.dem_path = dem_path
        try:
            self._state.dem_info = load_dem(dem_path)
        except Exception as exc:
            raise DesignFileError(f"The DEM could not be loaded: {exc}") from exc
        self._state.invalidate_flow_cache()
        self._state.baseline_result = None

    # ------------------------------------------------------------------ Site areas

    def _collect_areas(self):
        """The three site polygons as WKT, keyed for the design document."""
        return {key: geom for key in AREA_KEYS
                if (geom := self._area_geometry(key)) is not None}

    def _area_geometry(self, kind):
        """Dissolved geometry of the layer selected for *kind*, as WKT plus its CRS."""
        layer = {
            "boundary": self._panel.boundary_layer,
            "analysis": self._panel.analysis_area_layer,
            "earthworks": self._panel.earthworks_area_layer,
        }.get(kind)
        if layer is None:
            return None

        try:
            geometries = [f.geometry() for f in layer.getFeatures()
                          if f.hasGeometry() and not f.geometry().isEmpty()]
            if not geometries:
                return None
            merged = (geometries[0] if len(geometries) == 1
                      else QgsGeometry.unaryUnion(geometries))
            if merged is None or merged.isEmpty():
                return None
            return {"wkt": merged.asWkt(), "crs": layer.crs().authid() or None}
        except Exception as exc:
            print(f"TerrainFlow Assessment — could not read the {kind} area: {exc}")
            return None

    def _restore_areas(self, areas):
        """Rebuild the site polygons as real files and select them in the panel.

        Written to disk rather than held as memory layers because ``AnalysisWorker`` takes
        *paths* — a memory layer has no readable source, so baseline would silently run
        without the boundary it was handed.
        """
        for kind in AREA_KEYS:
            entry = areas.get(kind)
            if not entry:
                continue
            try:
                layer = self._layer_from_wkt(kind, entry["wkt"], entry.get("crs"))
            except Exception as exc:
                print(f"TerrainFlow Assessment — could not restore the {kind} area: {exc}")
                continue
            if layer is None:
                continue
            self._project.add_layer(layer)
            # Selecting it in the combo fires the panel's change signal, which is what
            # sets the corresponding path on the state — no need to set it here.
            self._panel.set_area_layer(kind, layer)

    def _layer_from_wkt(self, kind, wkt, crs_authid):
        """Write *wkt* to a GeoPackage in the session's output dir and load it."""
        geometry = QgsGeometry.fromWkt(wkt)
        if geometry is None or geometry.isEmpty():
            return None

        crs = QgsCoordinateReferenceSystem(crs_authid) if crs_authid else None
        if crs is None or not crs.isValid():
            # Fall back to the DEM's CRS: a design's areas are always in its terrain's.
            crs = self._dem_crs()

        scratch = QgsVectorLayer(
            f"Polygon?crs={crs.authid() if crs else ''}", f"{kind}_restored", "memory")
        feature = QgsFeature()
        feature.setGeometry(geometry)
        scratch.dataProvider().addFeatures([feature])
        scratch.updateExtents()

        target = os.path.join(self._state.output_dir, f"restored_{kind}.gpkg")
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.layerName = kind
        options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile

        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            scratch, target, self._project.transform_context(), options)
        # Signature varies across QGIS releases; the error code is always first.
        code = result[0] if isinstance(result, tuple) else result
        if code != QgsVectorFileWriter.NoError:
            raise DesignFileError(f"Could not write the {kind} area: {result}")

        written = QgsVectorLayer(f"{target}|layername={kind}", f"{kind.title()} (design)",
                                 "ogr")
        return written if written.isValid() else None

    def _dem_crs(self):
        info = self._state.dem_info
        wkt = getattr(info, "crs_wkt", None) if info is not None else None
        if not wkt:
            return None
        crs = QgsCoordinateReferenceSystem.fromWkt(wkt)
        return crs if crs.isValid() else None

    # ------------------------------------------------------------------ Prompts

    def _offer_baseline_rerun(self):
        """Offer the baseline run that restores scoring, and let the user decline it."""
        answer = QMessageBox.question(
            self._panel, "Run baseline analysis?",
            "Re-run the baseline analysis now to restore the design's scoring?\n\n"
            "Until it runs, the earthworks show their capacity but no water — the "
            "analysis rasters are recomputed rather than stored, so the design cannot "
            "be scored against a saved result.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.baseline_rerun_requested.emit()

    # ------------------------------------------------------------------ Messaging

    def _info(self, text):
        self._iface.messageBar().pushInfo("TerrainFlow Assessment", text)

    def _warn(self, text):
        self._iface.messageBar().pushWarning("TerrainFlow Assessment", text)

    def _critical(self, text):
        self._iface.messageBar().pushCritical("TerrainFlow Assessment", text)
