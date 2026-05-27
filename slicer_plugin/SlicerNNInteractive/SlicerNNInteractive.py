import io
import gzip
import requests
import copy
import threading
import time

import importlib.util

import numpy as np
from pathlib import Path
import os

import slicer
import qt
import vtk
from qt import QApplication, QPalette

from vtkmodules.util.numpy_support import vtk_to_numpy

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from PythonQt.QtGui import QMessageBox
from datetime import datetime
import json

###############################################################################
# Decorators and utility functions
###############################################################################


DEBUG_MODE = False


def debug_print(*args):
    if DEBUG_MODE:
        print(*args)


def ensure_synched(func):
    """
    Decorator that ensures the image and segment are synced before calling
    the actual prompt function.
    """

    def inner(self, *args, **kwargs):
        failed_to_sync = False

        if self.image_changed():
            debug_print(
                "Image changed (or not previously set). Calling upload_segment_to_server()"
            )
            result = self.upload_image_to_server()

            failed_to_sync = result is None

        if not failed_to_sync and self.selected_segment_changed():
            debug_print(
                "Segment changed (or not previously set). Calling upload_segment_to_server()"
            )
            self.remove_all_but_last_prompt()
            result = self.upload_segment_to_server()

            failed_to_sync = result is None
        else:
            debug_print("Segment did not change!")

        if not failed_to_sync:
            return func(self, *args, **kwargs)

    return inner


###############################################################################
# SlicerNNInteractive
###############################################################################


class SlicerNNInteractive(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)

        self.parent.title = _("nnInteractive")
        self.parent.categories = [
            translate("qSlicerAbstractCoreModule", "Segmentation")
        ]
        self.parent.dependencies = []  # List other modules if needed
        self.parent.contributors = ["Coen de Vente", "Kiran Vaidhya Venkadesh", "Bram van Ginneken", "Clara I. Sanchez"]
        self.parent.helpText = """
            This is an 3D Slicer extension for using nnInteractive.

            Read more about this plugin here: https://github.com/coendevente/SlicerNNInteractive.
            """
        self.parent.acknowledgementText = """When using SlicerNNInteractive, please cite as described here: https://github.com/coendevente/SlicerNNInteractive?tab=readme-ov-file#citation."""


###############################################################################
# SlicerNNInteractiveWidget
###############################################################################


class SlicerNNInteractiveWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    ###############################################################################
    # Setup and initialization functions
    ###############################################################################

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        # Add these initialization variables
        self.segmentation_history = []
        self.directory = None  # Will be set when directory is chosen (session, MR-xxx)
        self.base_directory = None  # Set after directory is set (go to the folder level of all subs)
        self._undo_redo_connected = False
        self.seg_directory = os.path.normpath("Z:/home/ext_xinwan/Bone_AI/tmp_data_seg")
        self.ai_seg_node = None
        self._last_volume_id = None
        self.review_session_dir = None  # Set when folder opened in review mode
        # Maps orientation label → (seg_node, ref_vol_node) populated by Duplicate button
        self.orientation_seg_map = {}

    def setup(self):
        """
        Overridden setup method. Initializes UI and setups up prompts.
        """
        ScriptedLoadableModuleWidget.setup(self)

        self.install_dependencies()
        safe_path = "Z:/home/ext_xinwan/SlicerNNInteractive_BT/slicer_plugin/SlicerNNInteractive/Resources/UI/SlicerNNInteractive.ui"
        if os.path.exists(safe_path):
            print(f"File exists at safe_path")
        ui_widget = slicer.util.loadUI(safe_path)
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self.scribble_segment_node_name = "ScribbleSegmentNode (do not touch)"

        # Set up editor widget
        self.ui.editor_widget.setMaximumNumberOfUndoStates(10)
        self.ui.editor_widget.setMRMLScene(slicer.mrmlScene)
        # Use the same segmentation parameter node as the Segment Editor core module
        segment_editor_singleton_tag = "SegmentEditor"
        self.segment_editor_node = slicer.mrmlScene.GetSingletonNode(segment_editor_singleton_tag, "vtkMRMLSegmentEditorNode")
        if self.segment_editor_node is None:
            self.segment_editor_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            self.segment_editor_node.UnRegister(None)
            self.segment_editor_node.SetSingletonTag(segment_editor_singleton_tag)
            self.segment_editor_node = slicer.mrmlScene.AddNode(self.segment_editor_node)
        self.ui.editor_widget.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.ui.editor_widget.setSegmentationNode(self.get_segmentation_node())

        # Set up style sheets for selected/unselected buttons
        self.selected_style = "background-color: #3498db; color: white"
        self.unselected_style = ""

        self.prompt_types = {
            "point": {
                "node_class": "vtkMRMLMarkupsFiducialNode",
                "node": None,
                "name": "PointPrompt",
                "display_node_markup_function": self.display_node_markup_point,
                "on_placed_function": self.on_point_placed,
                "button": self.ui.pbInteractionPoint,
                "button_text": self.ui.pbInteractionPoint.text,
                "button_icon_filename": "point_icon.svg",
            },
            "bbox": {
                "node_class": "vtkMRMLMarkupsROINode",
                "node": None,
                "name": "BBoxPrompt",
                "display_node_markup_function": self.display_node_markup_bbox,
                "on_placed_function": self.on_bbox_placed,
                "button": self.ui.pbInteractionBBox,
                "button_text": self.ui.pbInteractionBBox.text,
                "button_icon_filename": "bbox_icon.svg",
            },
            "lasso": {
                "node_class": "vtkMRMLMarkupsClosedCurveNode",
                "node": None,
                "name": "LassoPrompt",
                "display_node_markup_function": self.display_node_markup_lasso,
                "on_placed_function": self.on_lasso_placed,
                "button": self.ui.pbInteractionLasso,
                "button_text": self.ui.pbInteractionLasso.text,
                "button_icon_filename": "lasso_icon.svg",
            },
        }

        # Initialize contour checkbox state
        self.ui.contourCheckBox.setChecked(False) 

        self.setup_shortcuts()

        self.all_prompt_buttons = {}
        self.setup_prompts()

        self.init_ui_functionality()

        _ = self.get_current_segment_id()
        self.previous_states = {}

        # Add this at the end of your setup method:
        self.setup_auto_save()

        
    def init_ui_functionality(self):
        """
        Connect UI elements to functions.
        """
        self.ui.uploadProgressGroup.setVisible(False)

        # Load the saved server URL (default to an empty string if not set)
        savedServer = slicer.util.settingsValue("SlicerNNInteractive/server", "")
        self.ui.Server.text = savedServer
        self.server = savedServer.rstrip("/")

        self.ui.Server.editingFinished.connect(self.update_server)

        # Set initial prompt type
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)

        # Top buttons
        self.ui.pbResetSegment.clicked.connect(self.clear_current_segment)
        self.ui.pbNextSegment.clicked.connect(self.make_new_segment)

        # Connect Prompt Type buttons
        self.ui.pbPromptTypePositive.clicked.connect(
            self.on_prompt_type_positive_clicked
        )
        self.ui.pbPromptTypeNegative.clicked.connect(
            self.on_prompt_type_negative_clicked
        )

        self.ui.pbInteractionLassoCancel.setVisible(False)
        self.ui.pbInteractionScribble.clicked.connect(self.on_scribble_clicked)

        self.ui.pbInteractionLassoCancel.clicked.connect(self.on_lasso_cancel_clicked)

        # added connection for choosing scans
        self.ui.LoadScanButton.clicked.connect(self.loadScans)

        # PatientInfoBox checkbox shows/hides ClinicalInfoLabel
        self.ui.ClinicalInfoLabel.setVisible(False)
        self.ui.PatientInfoBox.toggled.connect(self.onPatientInfoToggled)

        # segSummaryLabel only visible when ShowSegCheckBox is checked
        self.ui.segSummaryLabel.setVisible(False)

        # Anatomy submit button
        self.ui.SubmitAnoButton.clicked.connect(self.on_submit_anatomy)

        # DiagnosisBox and anatomy panel hidden until review panel is shown
        self.ui.DiagnosisBox.setVisible(False)
        self.ui.groupBox_3.setVisible(False)
        self.ui.SubmitDiagButton.clicked.connect(self.on_submit_diagnosis)

        # added connection for reviewer panel
        self.ui.CorrectionButton.clicked.connect(self.checkReviewChoice)
        self.ui.RedoButton.clicked.connect(self.checkReviewChoice)
        # Totalseg segmentation controls
        self.ui.ShowSegCheckBox.toggled.connect(self.onShowSegToggled)

        # Restore saved seg directory (fall back to the default project path)
        _default_seg_dir = os.path.normpath("Z:/home/ext_xinwan/Bone_AI/tmp_data_totalseg")
        # savedSegDir = slicer.util.settingsValue("SlicerNNInteractive/seg_directory", _default_seg_dir)
        # if savedSegDir and os.path.exists(savedSegDir):
        self.seg_directory = _default_seg_dir

        # Observe active volume changes to auto-update AI seg
        self.addObserver(
            slicer.app.applicationLogic().GetSelectionNode(),
            vtk.vtkCommand.ModifiedEvent,
            self.on_active_volume_changed,
        )

        # Save the results
        self.ui.SaveButton.clicked.connect(self.saveResults)

        self.addObserver(slicer.app.applicationLogic().GetInteractionNode(), 
            slicer.vtkMRMLInteractionNode.InteractionModeChangedEvent, self.on_interaction_node_modified)
    
        self.ui.finalSaveButton.clicked.connect(self.on_final_save)

        # Add contour checkbox connection
        self.ui.contourCheckBox.stateChanged.connect(self.on_contour_checkbox_changed)

        # Isolate view checkbox
        self.ui.isolateViewCheckBox.setChecked(False)
        self.ui.isolateViewCheckBox.stateChanged.connect(self.on_isolate_view_changed)

        # Duplicate segmentation to other orientations
        self.ui.pbDuplicateToOrientations.clicked.connect(self.on_duplicate_to_orientations)

        # Gaussian smoothing of visible segments
        self.ui.pbGaussianSmooth.clicked.connect(self.on_gaussian_smooth_clicked)

        # Register segmentations to all images
        self.ui.pbFinalRegistration.clicked.connect(self.on_final_registration_clicked)

    
    def on_contour_checkbox_changed(self, state):
        """Handle contour checkbox state changes"""
        seg_node = self.get_segmentation_node()
        if not seg_node:
            return
        
        display_node = seg_node.GetDisplayNode()
        if not display_node:
            return
        
        # Get all segment IDs
        segmentation = seg_node.GetSegmentation()
        segment_ids = [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())]
        
        # Toggle fill opacity based on checkbox state
        fill_opacity = 0.0 if state else 1.0  # 0 for checked (contour only), 1 for unchecked (filled)
        
        with slicer.util.NodeModify(display_node):
            for segment_id in segment_ids:
                display_node.SetSegmentOpacity2DFill(segment_id, fill_opacity)
                segment = segmentation.GetSegment(segment_id)
                if segment:
                    segment.SetColor(1.0, 0.0, 0.0)

    def on_isolate_view_changed(self, state):
        """Show only the active segmentation and its reference volume when checked."""
        seg_node = self.get_segmentation_node()
        ref_vol = self.get_volume_node()

        all_seg_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        all_vol_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")

        if state:
            active_seg_id = seg_node.GetID() if seg_node else None
            ref_vol_id = ref_vol.GetID() if ref_vol else None
            for node in all_seg_nodes:
                node.GetDisplayNode().SetVisibility(node.GetID() == active_seg_id)
            for node in all_vol_nodes:
                dn = node.GetDisplayNode()
                if dn:
                    dn.SetVisibility(node.GetID() == ref_vol_id)
            # Set ref volume as background in all slice views so it is actually shown
            if ref_vol:
                ref_vol_id_str = ref_vol.GetID()
                layout_manager = slicer.app.layoutManager()
                for name in layout_manager.sliceViewNames():
                    logic = layout_manager.sliceWidget(name).sliceLogic()
                    logic.GetSliceCompositeNode().SetBackgroundVolumeID(ref_vol_id_str)
        else:
            for node in all_seg_nodes:
                node.GetDisplayNode().SetVisibility(True)
            for node in all_vol_nodes:
                dn = node.GetDisplayNode()
                if dn:
                    dn.SetVisibility(True)

    def get_volume_orientation(self, volume_node):
        """Return 'axial', 'coronal', or 'sagittal' for a volume node.

        The through-plane axis is the IJK axis with the largest voxel spacing
        (thickest slices / fewest slices).  We then look at which RAS direction
        that axis aligns with to decide the anatomical plane.
        RAS convention: 0=R/L → sagittal, 1=A/P → coronal, 2=S/I → axial.
        """
        spacing = volume_node.GetSpacing()  # (si, sj, sk)
        through_plane_axis = max(range(3), key=lambda i: spacing[i])

        # GetIJKToRASMatrix (4x4) accepts vtkMatrix4x4; includes spacing scale but
        # abs + argmax are unaffected since spacing is always positive.
        mat = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(mat)
        # Column `through_plane_axis` gives the RAS direction of that IJK axis
        tp_ras = [abs(mat.GetElement(r, through_plane_axis)) for r in range(3)]
        dominant = max(range(3), key=lambda i: tp_ras[i])

        return ["sagittal", "coronal", "axial"][dominant]

    def on_duplicate_to_orientations(self):
        """Resample the current segmentation into one target volume per distinct non-reference orientation."""
        import SimpleITK as sitk
        import sitkUtils

        seg_node = self.get_segmentation_node()
        if not seg_node:
            slicer.util.warningDisplay("No segmentation found to duplicate.")
            return

        ref_volume = self.get_volume_node()
        if not ref_volume:
            slicer.util.warningDisplay("No reference volume found.")
            return

        ref_orientation = self.get_volume_orientation(ref_volume)

        # Reset the map and seed with the original segmentation.
        # Tuple: (seg_node, ref_vol, iso_labelmap_node | None)
        # iso_labelmap_node is the isotropic intermediate labelmap kept alive for
        # on_final_registration_clicked to use as export reference.
        self.orientation_seg_map = {ref_orientation: (seg_node, ref_volume, None)}

        # Collect one representative volume per distinct orientation (excluding reference)
        all_volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        seen_orientations = {ref_orientation}
        target_volumes = []  # list of (volume_node, orientation_label)
        for vol in all_volumes:
            if vol.GetID() == ref_volume.GetID():
                continue
            orient = self.get_volume_orientation(vol)
            if orient not in seen_orientations:
                seen_orientations.add(orient)
                target_volumes.append((vol, orient))

        if not target_volumes:
            slicer.util.warningDisplay(
                "No volumes with a different orientation were found.\n"
                "Load axial or coronal volumes before duplicating."
            )
            return

        # Export current segmentation to a temporary labelmap in reference volume space
        tmp_labelmap = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "__tmp_dup_labelmap__"
        )
        try:
            slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
                seg_node, tmp_labelmap, ref_volume
            )
            sitk_seg = sitkUtils.PullVolumeFromSlicer(tmp_labelmap)
        finally:
            slicer.mrmlScene.RemoveNode(tmp_labelmap)

        created = []
        for target_vol, orient in target_volumes:
            # Resample ref seg to the target volume's native spacing/size.
            sitk_target = sitkUtils.PullVolumeFromSlicer(target_vol)

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(sitk_target.GetSpacing())
            resampler.SetSize(sitk_target.GetSize())
            resampler.SetOutputDirection(sitk_target.GetDirection())
            resampler.SetOutputOrigin(sitk_target.GetOrigin())
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(0)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampled = resampler.Execute(sitk_seg)

            # Import resampled labelmap as a new segmentation node
            new_labelmap = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", f"__tmp_dup_{orient}__"
            )
            try:
                sitkUtils.PushVolumeToSlicer(resampled, new_labelmap)
                new_seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                new_seg_node.SetName(f"{seg_node.GetName()}_{orient}")
                new_seg_node.SetReferenceImageGeometryParameterFromVolumeNode(target_vol)
                new_seg_node.CreateDefaultDisplayNodes()
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                    new_labelmap, new_seg_node
                )
                self.orientation_seg_map[orient] = (new_seg_node, target_vol, None)
            finally:
                slicer.mrmlScene.RemoveNode(new_labelmap)

            created.append(orient)
            print(f"[CHECK] duplicated {orient} seg spacing: {resampled.GetSpacing()}")

        # Record in history
        if self.directory:
            entry = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'action': "duplicate_to_orientations",
                'orientations_created': created,
                'reference_volume': ref_volume.GetName() if ref_volume else "unknown",
                'is_final': False,
                'filename': None,
                'prompt_type': None,
                'is_reset': False,
            }
            self.segmentation_history.append(entry)
            self.save_history_file()

        slicer.util.infoDisplay(
            f"Duplicated segmentation to: {', '.join(created)}."
        )

    # ------------------------------------------------------------------
    # Gaussian smoothing
    # ------------------------------------------------------------------

    def on_gaussian_smooth_clicked(self):
        """Apply Gaussian smoothing (σ=1 mm) to every visible segment in-place."""
        import SimpleITK as sitk
        import sitkUtils

        seg_node = self.get_segmentation_node()
        if not seg_node:
            slicer.util.warningDisplay("No segmentation found.")
            return

        volume_node = self.get_volume_node()
        segmentation = seg_node.GetSegmentation()
        display_node = seg_node.GetDisplayNode()
        sigma_mm = 1.0

        for i in range(segmentation.GetNumberOfSegments()):
            seg_id = segmentation.GetNthSegmentID(i)
            if display_node and not display_node.GetSegmentVisibility(seg_id):
                continue

            segment = segmentation.GetSegment(seg_id)
            seg_name = segment.GetName()
            seg_color = list(segment.GetColor())

            # Export single segment to a temporary labelmap
            tmp_lm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            try:
                seg_ids_vtk = vtk.vtkStringArray()
                seg_ids_vtk.InsertNextValue(seg_id)
                slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                    seg_node, seg_ids_vtk, tmp_lm, volume_node
                )

                sitk_img = sitkUtils.PullVolumeFromSlicer(tmp_lm)
                sitk_float = sitk.Cast(sitk_img, sitk.sitkFloat32)
                smoothed = sitk.SmoothingRecursiveGaussian(sitk_float, sigma=sigma_mm)
                binary = sitk.BinaryThreshold(
                    smoothed, lowerThreshold=0.5, upperThreshold=float("inf"),
                    insideValue=1, outsideValue=0
                )
                binary = sitk.Cast(binary, sitk.sitkUInt8)
                sitkUtils.PushVolumeToSlicer(binary, tmp_lm)

                # Replace the old segment: remove it, import the smoothed labelmap,
                # then restore name and colour on the newly created segment.
                segmentation.RemoveSegment(seg_id)
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                    tmp_lm, seg_node
                )
                new_id = segmentation.GetNthSegmentID(segmentation.GetNumberOfSegments() - 1)
                new_seg = segmentation.GetSegment(new_id)
                if new_seg:
                    new_seg.SetName(seg_name)
                    new_seg.SetColor(*seg_color)
                    # After smoothing, show contour only (fill opacity = 0)
                    if display_node:
                        display_node.SetSegmentOpacity2DFill(new_id, 0.0)
            finally:
                slicer.mrmlScene.RemoveNode(tmp_lm)

        # Sync the contour checkbox to reflect the new state
        self.ui.contourCheckBox.blockSignals(True)
        self.ui.contourCheckBox.setChecked(True)
        self.ui.contourCheckBox.blockSignals(False)

    # ------------------------------------------------------------------
    # Final registration — resample orientation segmentations to every image
    # ------------------------------------------------------------------

    def on_final_registration_clicked(self):
        """Resample each orientation segmentation to every loaded image of that orientation.

        After this call, every volume node has a corresponding segmentation node
        named  <volume_name>_seg  in the scene (isotropic spacing).
        The mapping is stored in  self.registered_seg_nodes  for later saving.
        """
        import SimpleITK as sitk
        import sitkUtils

        if not self.orientation_seg_map:
            slicer.util.warningDisplay(
                "No orientation segmentation map found.\n"
                "Please click 'Duplicate segmentation to other orientations' first."
            )
            return

        all_volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")

        self.registered_seg_nodes = []  # [(seg_node, vol_node)]

        for vol in all_volumes:
            orient = self.get_volume_orientation(vol)
            if orient not in self.orientation_seg_map:
                continue  # no segmentation for this orientation

            src_seg_node, src_ref_vol, *_ = self.orientation_seg_map[orient]

            # Export source segmentation to a temporary labelmap in its reference space.
            tmp_src = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "__tmp_finalreg")
            try:
                slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
                    src_seg_node, tmp_src, src_ref_vol
                )
                sitk_src = sitkUtils.PullVolumeFromSlicer(tmp_src)
            finally:
                slicer.mrmlScene.RemoveNode(tmp_src)

            # Resample to the native spacing/size of the target volume
            sitk_vol = sitkUtils.PullVolumeFromSlicer(vol)

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(sitk_vol.GetSpacing())
            resampler.SetSize(sitk_vol.GetSize())
            resampler.SetOutputDirection(sitk_vol.GetDirection())
            resampler.SetOutputOrigin(sitk_vol.GetOrigin())
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(0)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampled = resampler.Execute(sitk_src)

            tmp_dst = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            try:
                sitkUtils.PushVolumeToSlicer(resampled, tmp_dst)
                new_seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                new_seg.SetName(f"{vol.GetName()}_seg")
                new_seg.SetReferenceImageGeometryParameterFromVolumeNode(vol)
                new_seg.CreateDefaultDisplayNodes()
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                    tmp_dst, new_seg
                )
            finally:
                slicer.mrmlScene.RemoveNode(tmp_dst)

            # Spacing check: output must match the reference volume exactly
            out_sp = resampled.GetSpacing()
            out_sz = resampled.GetSize()
            ref_sp = sitk_vol.GetSpacing()
            ref_sz = sitk_vol.GetSize()
            sp_ok = all(abs(out_sp[i] - ref_sp[i]) < 1e-3 for i in range(3))
            sz_ok = out_sz == ref_sz
            status = "OK" if (sp_ok and sz_ok) else "MISMATCH"
            print(f"[CHECK] {vol.GetName()}_seg vs ref vol [{status}] "
                  f"spacing: {out_sp} vs {ref_sp} | size: {out_sz} vs {ref_sz}")

            self.registered_seg_nodes.append((new_seg, vol))

        # Record in history
        if self.directory:
            reg_entry = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'action': "register_to_all_images",
                'num_registered': len(self.registered_seg_nodes),
                'volumes': [v.GetName() for _, v in self.registered_seg_nodes],
                'is_final': False,
                'filename': None,
                'prompt_type': None,
                'is_reset': False,
            }
            self.segmentation_history.append(reg_entry)
            self.save_history_file()

        n = len(self.registered_seg_nodes)
        slicer.util.infoDisplay(
            f"Registration complete: {n} segmentation(s) now cover all loaded images."
        )

    # ------------------------------------------------------------------
    # Per-image save helper (called from on_final_save)
    # ------------------------------------------------------------------

    def save_registered_segs_to_image_folders(self, review_mode=None):
        """Save every registered segmentation to segs/.

        - First-time save:  segmentation_history/segs/
        - Review save:      review/<timestamp>/segs/

        Each <vol_name>_seg node was already resampled to the native spacing of
        its reference volume during registration, so we just export it directly.
        Files are named <volume_name>_seg.nii.gz.
        """
        import SimpleITK as sitk
        import sitkUtils
        import re

        if not self.directory:
            print("No session directory set; cannot save registered segmentations.")
            return

        in_review = review_mode and self.review_session_dir
        if in_review:
            segs_dir = os.path.join(self.review_session_dir, "segs")
        else:
            segs_dir = os.path.join(self.directory, "segmentation_history", "segs")
        os.makedirs(segs_dir, exist_ok=True)

        has_registered = bool(getattr(self, "registered_seg_nodes", None))
        if not has_registered:
            return  # nothing to save
        pairs = self.registered_seg_nodes

        saved = []
        for seg_node, vol_node in pairs:
            vol_name_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', vol_node.GetName())
            out_path = os.path.join(segs_dir, f"{vol_name_clean}_seg.nii.gz")

            seg = seg_node.GetSegmentation()
            seg_ids = [seg.GetNthSegmentID(i) for i in range(seg.GetNumberOfSegments())]
            tmp_lm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            try:
                slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                    seg_node, seg_ids, tmp_lm, vol_node
                )
                sitk_lm = sitkUtils.PullVolumeFromSlicer(tmp_lm)
                sitk.WriteImage(sitk_lm, out_path)
                saved.append(out_path)
            except Exception as e:
                print(f"Failed to save {out_path}: {e}")
            finally:
                slicer.mrmlScene.RemoveNode(tmp_lm)

        if saved:
            print(f"Saved {len(saved)} segmentation(s) to {segs_dir}:")
            for p in saved:
                print(f"  {p}")

    def setup_auto_save(self):
        """Initialize auto-save functionality"""
        
        # Add observer for undo events
        self.connect_undo_redo_buttons()
    
    def connect_undo_redo_buttons(self):
        """Connect to the actual undo/redo buttons in segment editor"""
        editor = self.ui.editor_widget
        undo_button = editor.findChild("QToolButton", "UndoButton")
        redo_button = editor.findChild("QToolButton", "RedoButton")
        
        if undo_button:
            undo_button.clicked.connect(self.on_undo_action)
        if redo_button:
            redo_button.clicked.connect(self.on_redo_action)
    
    def save_segmentation_nii(self, action_type, prompt_type=None, is_final=False, isotropic=False):
        """Save current segmentation as NIfTI.gz"""
        if not self.directory:
            return None
        
        # Get ref volume name
        volume_node = self.get_volume_node()
        volume_name = volume_node.GetName() if volume_node else "unknown_volume"

        # Clean volume name for filename
        import re
        volume_name_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', volume_name)
        
        # Determine save directory based on review mode
        in_review = self.ui.CorrectionButton.isChecked() or self.ui.RedoButton.isChecked()
        if in_review and self.review_session_dir:
            save_dir = self.review_session_dir
        else:
            save_dir = os.path.join(self.directory, "segmentation_history")
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate appropriate filename
        if is_final:
            filename = f"FINAL_{volume_name_clean}_{timestamp}.nii.gz"
        else:
            prefix = prompt_type if prompt_type else action_type
            filename = f"{prefix}_{volume_name_clean}_{timestamp}.nii.gz"

        filepath = os.path.join(save_dir, filename)
        
        seg_node = self.get_segmentation_node()
        if not seg_node:
            return None
        
        # Create temporary labelmap node
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        storage_node = None
        try:
            # Export to labelmap using reference volume geometry
            slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
                seg_node,
                labelmap_node,
                self.get_volume_node()
            )

            if isotropic:
                # Resample to isotropic spacing (min of original spacings) using SimpleITK
                import SimpleITK as sitk
                import sitkUtils
                sitk_image = sitkUtils.PullVolumeFromSlicer(labelmap_node)
                orig_spacing = sitk_image.GetSpacing()
                min_sp = min(orig_spacing)
                new_spacing = [min_sp] * 3
                orig_size = sitk_image.GetSize()
                new_size = [int(round(orig_size[i] * orig_spacing[i] / min_sp)) for i in range(3)]
                resampler = sitk.ResampleImageFilter()
                resampler.SetOutputSpacing(new_spacing)
                resampler.SetSize(new_size)
                resampler.SetOutputDirection(sitk_image.GetDirection())
                resampler.SetOutputOrigin(sitk_image.GetOrigin())
                resampler.SetTransform(sitk.Transform())
                resampler.SetDefaultPixelValue(0)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampled = resampler.Execute(sitk_image)
                sitk.WriteImage(resampled, filepath)
            else:
                # Create storage node and write at reference volume spacing
                storage_node = labelmap_node.CreateDefaultStorageNode()
                slicer.mrmlScene.AddNode(storage_node)
                storage_node.SetFileName(filepath)
                if not storage_node.WriteData(labelmap_node):
                    raise RuntimeError(f"Failed to save segmentation to {filepath}")

            # Verify file was created
            if not os.path.exists(filepath):
                raise RuntimeError(f"Output file not created: {filepath}")
                        
            # Record in history
            history_entry = {
                'timestamp': timestamp,
                'filename': filename,
                'action': action_type,
                'prompt_type': prompt_type,
                'is_reset':action_type == "reset",
                'is_final': is_final,
                'reference_volume': volume_name
            }

            if in_review:
                self.save_review_history(history_entry)
            else:
                if not any(entry['timestamp'] == timestamp for entry in self.segmentation_history):
                    self.segmentation_history.append(history_entry)
                self.save_history_file()
            
            return filepath
        
        except Exception as e:
            debug_print(f"Error saving segmentation: {str(e)}")
            # Remove partially written file if it exists
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return None
            
        finally:
            # Clean up temporary nodes
            if storage_node:
                slicer.mrmlScene.RemoveNode(storage_node)
            if labelmap_node:
                slicer.mrmlScene.RemoveNode(labelmap_node)

    def save_history_file(self):
        """Save history to JSON file"""
        if not self.directory:
            return

        history_dir = os.path.join(self.directory, "segmentation_history")
        os.makedirs(history_dir, exist_ok=True)
        history_path = os.path.join(history_dir, "history.json")
            
        # Initialize with empty list if no history exists yet
        current_history = []

        # Load existing history if file exists
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    current_history = json.load(f)
            except Exception as e:
                debug_print(f"Error reading history file: {e}")
                current_history = []

        # Add new entries that aren't already in the history
        new_entries = []
        for entry in self.segmentation_history:
            # Check if entry already exists in history (based on timestamp)
            if not any(e.get('timestamp') == entry.get('timestamp') for e in current_history):
                new_entries.append(entry)
        
        # Combine old and new entries
        updated_history = current_history + new_entries
        
        # Save the combined history
        try:
            with open(history_path, 'w') as f:
                json.dump(updated_history, f, indent=2)
        except Exception as e:
            debug_print(f"Error saving history file: {e}")

    def get_review_history_path(self):
        """Return path to history.json inside the current review session directory."""
        if not self.review_session_dir:
            return None
        os.makedirs(self.review_session_dir, exist_ok=True)
        return os.path.join(self.review_session_dir, "history.json")

    def save_review_history(self, entry):
        """Append an entry to the review session history file."""
        history_path = self.get_review_history_path()
        if not history_path:
            return

        existing_history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    existing_history = json.load(f)
            except Exception as e:
                debug_print(f"Error reading review history: {e}")

        existing_history.append(entry)

        try:
            with open(history_path, 'w') as f:
                json.dump(existing_history, f, indent=2)
        except Exception as e:
            debug_print(f"Error saving review history: {e}")

    def on_undo_action(self):
        """Called when undo button is clicked"""
        # Let the undo complete first
        qt.QTimer.singleShot(100, lambda: self.save_segmentation_nii("undo"))

    def on_redo_action(self):
        """Called when redo button is clicked"""
        # Let the redo complete first  
        qt.QTimer.singleShot(100, lambda: self.save_segmentation_nii("redo"))

    def on_final_save(self):
        """Handle final save button click.

        Two-step save:
          1. Save all main orientation segmentations to history.
          2. Save every registered segmentation to its image folder at the
             native (non-isotropic) spacing of the corresponding volume.
        """
        if not self.directory:
            slicer.util.warningDisplay("Please set a directory first", windowTitle="Save Error")
            return

        # --- Step 1: save main segmentations to history/review dir ---
        in_review = self.ui.CorrectionButton.isChecked() or self.ui.RedoButton.isChecked()
        has_registered = bool(getattr(self, "registered_seg_nodes", None))

        if self.orientation_seg_map:
            # Multi-orientation workflow: save one file per orientation.
            saved_results = []
            original_vol = self.get_volume_node()
            original_seg = self.get_segmentation_node()
            for orient, (seg_node, ref_vol, *_) in self.orientation_seg_map.items():
                self.ui.editor_widget.setSegmentationNode(seg_node)
                self.ui.editor_widget.setSourceVolumeNode(ref_vol)
                r = self.save_segmentation_nii(f"FINAL_{orient}", is_final=True, isotropic=False)
                if r:
                    saved_results.append(r)
            self.ui.editor_widget.setSegmentationNode(original_seg)
            self.ui.editor_widget.setSourceVolumeNode(original_vol)
            result = saved_results[0] if saved_results else None
        elif in_review and has_registered:
            # Review mode: write each seg directly to review_session_dir.
            # Bypass save_segmentation_nii — it uses CorrectionButton/RedoButton to
            # detect review mode (not set here) and ExportVisibleSegments (segs are hidden).
            import SimpleITK as sitk
            import sitkUtils
            import re as _re
            saved_results = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for seg_node, ref_vol in self.registered_seg_nodes:
                vol_name = ref_vol.GetName() if ref_vol else "unknown"
                vol_name_clean = _re.sub(r'[^a-zA-Z0-9_-]', '_', vol_name)
                out_path = os.path.join(self.review_session_dir,
                                        f"FINAL_{vol_name_clean}_{timestamp}.nii.gz")
                seg = seg_node.GetSegmentation()
                seg_ids = [seg.GetNthSegmentID(i) for i in range(seg.GetNumberOfSegments())]
                tmp_lm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                try:
                    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                        seg_node, seg_ids, tmp_lm, ref_vol
                    )
                    sitk_lm = sitkUtils.PullVolumeFromSlicer(tmp_lm)
                    sitk.WriteImage(sitk_lm, out_path)
                    saved_results.append(out_path)
                except Exception as e:
                    print(f"Failed to save review FINAL for {vol_name}: {e}")
                finally:
                    slicer.mrmlScene.RemoveNode(tmp_lm)
            result = saved_results[0] if saved_results else None
        else:
            result = self.save_segmentation_nii("FINAL", is_final=True, isotropic=True)

        if result:
            if self.ui.CorrectionButton.isChecked():
                review_mode = "correction"
                message = "Final corrected segmentation saved"
            elif self.ui.RedoButton.isChecked():
                review_mode = "redo"
                message = "Final re-segmentation saved"
            else:
                review_mode = None
                message = "Final segmentation saved"

            if review_mode:
                completion_entry = {
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'action': "review_complete",
                    'review_mode': review_mode,
                    'filename': os.path.basename(result),
                    'notes': f"Completed {review_mode} review"
                }
                self.save_review_history(completion_entry)
            else:
                self.update_history_as_final(result)

            # --- Step 2: save per-image segmentations to segs folder ---
            self.save_registered_segs_to_image_folders(review_mode=review_mode)

            slicer.util.infoDisplay(f"{message}:\n{result}", windowTitle="Save Successful")
        else:
            slicer.util.errorDisplay("Failed to save final segmentation", windowTitle="Save Error")
    
    def update_history_as_final(self, final_filepath):
        """Mark all previous entries as not-final and update the final one"""
        filename = os.path.basename(final_filepath)
        
        # Update all entries in history
        for entry in self.segmentation_history:
            entry['is_final'] = False
            if entry['filename'] == filename:
                entry['is_final'] = True
        
        self.save_history_file()
    
    def check_existing_history(self):
        """Check if history exists and return FINAL segmentation path if available"""
        if not self.directory:
            return None, []
            
        history_dir = os.path.join(self.directory, "segmentation_history")
        history_file = os.path.join(history_dir, "history.json")
        
        if not os.path.exists(history_file):
            return None, []
            
        try:
            with open(history_file, 'r') as f:
                existing_history = json.load(f)
                
            # Find the most recent FINAL segmentation
            final_entry = next((e for e in reversed(existing_history) if e.get('is_final')), None)
            final_path = os.path.join(history_dir, final_entry['filename']) if final_entry else None
            
            # Merge with any in-memory history
            if hasattr(self, 'segmentation_history'):
                # Filter out duplicates
                new_entries = [e for e in self.segmentation_history 
                            if not any(ex.get('timestamp') == e.get('timestamp') for ex in existing_history)]
                existing_history.extend(new_entries)
            
            return final_path, existing_history
        except Exception as e:
            debug_print(f"Error reading history: {e}")
            return None, []
    
    def checkReviewChoice(self):
        if self.ui.CorrectionButton.isChecked() and (not self.ui.RedoButton.isChecked()):
            correction_entry = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'action': "correction_start",
                'review_mode': "correction",
                'notes': "Beginning manual corrections"
            }
            self.save_review_history(correction_entry)
            slicer.util.infoDisplay("Now recording manual correction review session", windowTitle="Correction Mode")

        elif not self.ui.CorrectionButton.isChecked() and self.ui.RedoButton.isChecked():
            # Redo mode — hide existing segmentations so reviewer starts fresh
            seg_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
            for seg_node in seg_nodes:
                if seg_node.GetName() != self.scribble_segment_node_name:
                    seg_node.GetDisplayNode().SetVisibility(False)

            self.get_segmentation_node()

            redo_entry = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'action': "redo_start",
                'review_mode': "redo",
                'notes': "Beginning re-segmentation"
            }
            self.save_review_history(redo_entry)
            slicer.util.infoDisplay("Re-segmentation mode activated", windowTitle="Redo Mode")

        elif self.ui.RedoButton.isChecked() and self.ui.CorrectionButton.isChecked():
            slicer.util.errorDisplay("Only one option can be selected", windowTitle="Input Error")

            

    def setup_shortcuts(self):
        """
        Sets up keyboard shortcuts.
        """
        shortcuts = {
            "o": self.ui.pbInteractionPoint.click,
            "b": self.ui.pbInteractionBBox.click,
            "l": self.ui.pbInteractionLasso.click,
            "s": self.ui.pbInteractionScribble.click,
            "e": self.make_new_segment,
            "r": self.clear_current_segment,
            "Shift+L": self.submit_lasso_if_present,
            "t": self.toggle_prompt_type,  # Add 'T' shortcut to toggle between positive/negative
        }
        self.shortcut_items = {}

        for shortcut_key, shortcut_event in shortcuts.items():
            debug_print(f"Added shortcut for {shortcut_key}: {shortcut_event}")
            shortcut = qt.QShortcut(
                qt.QKeySequence(shortcut_key), slicer.util.mainWindow()
            )
            shortcut.activated.connect(shortcut_event)
            self.shortcut_items[shortcut_key] = shortcut

    def setup_dataparameters(self):
        self.base_directory = None
        self.directory = None
        self.seg_directory = None
        self.ai_seg_node = None
        self._last_volume_id = None


    def remove_shortcut_items(self):
        """
        Called at cleanup to remove all the shortcuts we attached.
        """
        if hasattr(self, "shortcut_items"):
            for _, shortcut in self.shortcut_items.items():
                shortcut.setParent(None)
                shortcut.deleteLater()
                shortcut = None

    def install_dependencies(self):
        """
        Checks for (and installs if needed) python packages needed by the module.
        """
        dependencies = {
            "requests_toolbelt": "requests_toolbelt",
            "skimage": "scikit-image",
            "pandas":"pandas"
        }

        for dependency in dependencies:
            if self.check_dependency_installed(dependency, dependencies[dependency]):
                continue
            self.run_with_progress_bar(
                self.pip_install_wrapper,
                (dependencies[dependency],),
                "Installing dependencies: %s" % dependency,
            )

    def check_dependency_installed(self, import_name, module_name_and_version):
        """
        Checks if a package is installed with the correct version.
        """
        if "==" in module_name_and_version:
            module_name, module_version = module_name_and_version.split("==")
        else:
            module_name = module_name_and_version
            module_version = None

        spec = importlib.util.find_spec(import_name)
        if spec is None:
            # Not installed
            return False

        if module_version is not None:
            import importlib.metadata as metadata
            try:
                version = metadata.version(module_name)
                if version != module_version:
                    # Version mismatch
                    return False
            except metadata.PackageNotFoundError:
                debug_print(f"Could not determine version for {module_name}.")

        return True

    def pip_install_wrapper(self, command, event):
        """
        Installs pip packages.
        """
        slicer.util.pip_install(command)
        event.set()

    def run_with_progress_bar(self, target, args, title):
        """
        Runs a function in a background thread, while showing a progress bar in the UI
        as a pop up window.
        """
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 100
        self.progressbar.setLabelText(title)

        parallel_event = threading.Event()
        dep_thread = threading.Thread(
            target=target,
            args=(
                *args,
                parallel_event,
            ),
        )
        dep_thread.start()
        while not parallel_event.is_set():
            slicer.app.processEvents()
        dep_thread.join()

        self.progressbar.close()

    def cleanup(self):
        """
        Clean up resources when the module is closed.
        """
        # Disconnect undo/redo buttons
        editor = self.ui.editor_widget
        undo_button = editor.findChild("QToolButton", "UndoButton")
        redo_button = editor.findChild("QToolButton", "RedoButton")
        
        if undo_button:
            try:
                undo_button.clicked.disconnect(self.on_undo_action)
            except:
                pass
        if redo_button:
            try:
                redo_button.clicked.disconnect(self.on_redo_action)
            except:
                pass

        self.removeObservers()

        if hasattr(self, "_qt_event_filters"):
            for slice_view, event_filter in self._qt_event_filters:
                slice_view.removeEventFilter(event_filter)
            self._qt_event_filters = []

        self.remove_shortcut_items()

        """Clean up any pending load entry"""
        if hasattr(self, 'pending_load_entry'):
            del self.pending_load_entry

    def __del__(self):
        """
        Called when the widget is destroyed.
        """
        self.remove_shortcut_items()

    ###############################################################################
    # Prompt and markup setup functions
    ###############################################################################

    def setup_prompts(self, skip_if_exists=False):
        if not skip_if_exists:
            self.remove_prompt_nodes()

        for prompt_name, prompt_type in self.prompt_types.items():
            if skip_if_exists and slicer.mrmlScene.GetFirstNodeByName(
                prompt_type["name"]
            ):
                debug_print("Skipping", prompt_name)
                continue
            node = slicer.mrmlScene.AddNewNodeByClass(prompt_type["node_class"])
            node.SetName(prompt_type["name"])
            node.CreateDefaultDisplayNodes()

            display_node = node.GetDisplayNode()
            prompt_type["display_node_markup_function"](display_node)

            prompt_type["button"].setStyleSheet(
                f"""
                QPushButton {{
                    {self.unselected_style}
                }}
                QPushButton:checked {{
                    {self.selected_style}
                }}
            """
            )

            self.prev_caller = None

            if prompt_type["on_placed_function"] is not None:
                node.AddObserver(
                    slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                    prompt_type["on_placed_function"],
                )

            prompt_type["node"] = node
            prompt_type["button"].clicked.connect(lambda checked, prompt_name=prompt_name: self.on_place_button_clicked(checked, prompt_name)) 
            self.all_prompt_buttons[prompt_name] = prompt_type["button"]

            light_dark_mode = self.is_ui_dark_or_light_mode()
            icon = qt.QIcon(self.resourcePath(f"Icons/prompts/{light_dark_mode}/{prompt_type['button_icon_filename']}"))
            prompt_type["button"].setIcon(icon)

        if (
            not skip_if_exists
            or slicer.mrmlScene.GetFirstNodeByName(self.scribble_segment_node_name)
            is None
        ):
            self.setup_scribble_prompt()

            self.ui.pbInteractionScribble.setStyleSheet(
                f"""
                QPushButton {{
                    {self.unselected_style}
                }}
                QPushButton:checked {{
                    {self.selected_style}
                }}
            """
            )
            self.all_prompt_buttons["scribble"] = self.ui.pbInteractionScribble

        # To make sure that when segment is reset, no interaction is selected (without this code
        # the last interaction tool gets selected)
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

    def setup_scribble_prompt(self):
        """
        Creates a hidden "Segment Editor" for the scribble prompt.
        """
        import qSlicerSegmentationsModuleWidgetsPythonQt

        # Create a background (headless) segment editor
        self.scribble_editor_widget = (
            qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        )
        self.scribble_editor_widget.setMRMLScene(slicer.mrmlScene)
        self.scribble_editor_widget.setMaximumNumberOfUndoStates(10)

        # Create a separate SegmentEditorNode
        self.scribble_editor_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentEditorNode"
        )
        self.scribble_editor_widget.setMRMLSegmentEditorNode(self.scribble_editor_node)

        self.scribble_segment_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode"
        )
        self.scribble_segment_node.SetReferenceImageGeometryParameterFromVolumeNode(
            self.get_volume_node()
        )
        self.scribble_segment_node.SetName(self.scribble_segment_node_name)

        # Make sure the node exists and is set
        self.scribble_editor_widget.setSegmentationNode(self.scribble_segment_node)

        self.scribble_segment_node.CreateDefaultDisplayNodes()
        self.scribble_segment_node.GetSegmentation().AddEmptySegment(
            "bg", "bg", [0.0, 0.0, 1.0]
        )
        self.scribble_segment_node.GetSegmentation().AddEmptySegment(
            "fg", "fg", [0.0, 0.0, 1.0]
        )
        dn = self.scribble_segment_node.GetDisplayNode()

        opacity = 0.2
        dn.SetSegmentOpacity2DFill("bg", opacity)
        dn.SetSegmentOpacity2DOutline("bg", opacity)
        dn.SetSegmentOpacity2DFill("fg", opacity)
        dn.SetSegmentOpacity2DOutline("fg", opacity)

        self._prev_scribble_mask = None
            
        light_dark_mode = self.is_ui_dark_or_light_mode()
        icon = qt.QIcon(self.resourcePath(f"Icons/prompts/{light_dark_mode}/scribble_icon.svg"))
        self.ui.pbInteractionScribble.setIcon(icon)

    def is_ui_dark_or_light_mode(self):
        # Returns whether the current appearance of the UI is dark mode (will return "dark")
        # or light mode (will return "light")
        current_style = slicer.app.settings().value("Styles/Style")

        if current_style == "Dark Slicer":
            return "dark"
        elif current_style == "Light Slicer":
            return "light"
        elif current_style == "Slicer":
            app_palette = QApplication.instance().palette()
            window_color = app_palette.color(QPalette.Active, QPalette.Window)
            lightness = window_color.lightness()
            dark_mode_threshold = 128

            if lightness < dark_mode_threshold:
                return "dark"
            else:
                return "light"
        return "light"

    def remove_prompt_nodes(self):
        """
        Removes all the Markups/Fiducials prompts.
        """

        def _remove(node_name):
            existing_nodes = slicer.mrmlScene.GetNodesByName(node_name)
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)
                    slicer.mrmlScene.RemoveNode(node)

        for prompt_type in list(self.prompt_types.values()):
            _remove(prompt_type["name"])

        self.ui.pbInteractionLassoCancel.setVisible(False)

        # Remove scribble observer before destroying the node, otherwise removing the
        # node from the scene fires AnyEvent and spuriously triggers on_scribble_finished.
        if hasattr(self, "_scribble_labelmap_callback_tag") and hasattr(self, "scribble_segment_node"):
            tag = self._scribble_labelmap_callback_tag.get("tag", None)
            if tag:
                self.scribble_segment_node.RemoveObserver(tag)
            del self._scribble_labelmap_callback_tag

        _remove(self.scribble_segment_node_name)

    def on_interaction_node_modified(self, caller, event):
        """
        Deselect prompt button if interaction mode is not place point anymore
        """

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        for prompt_type in self.prompt_types.values():
            if interactionNode.GetCurrentInteractionMode() != slicer.vtkMRMLInteractionNode.Place:
                if prompt_type["name"] == "LassoPrompt" and (self.ui.pbInteractionLasso.isChecked()):
                    self.submit_lasso_if_present()
                prompt_type["button"].setChecked(False)
            elif interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.Place:
                placingThisNode = (selectionNode.GetActivePlaceNodeID() == prompt_type["node"].GetID())
                prompt_type["button"].setChecked(placingThisNode)

        # Stop scribble if placing markup
        if interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.Place:
            self.ui.pbInteractionScribble.setChecked(False)

    def remove_all_but_last_prompt(self):
        """
        Removes all but the most recently placed markup points
        (helpful when segment change was detected).
        """
        last_modified_node = None
        all_nodes = []

        for prompt_type in self.prompt_types.values():
            existing_nodes = slicer.mrmlScene.GetNodesByName(prompt_type["name"])
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)

                    all_nodes.append(node)
                    if (
                        last_modified_node is None
                        or node.GetMTime() > last_modified_node.GetMTime()
                    ):
                        last_modified_node = node

        for node in all_nodes:
            n = node.GetNumberOfControlPoints()

            if node == last_modified_node:
                if node.GetName() == "LassoPrompt":
                    continue
                n -= 1

            for i in range(n):
                node.RemoveNthControlPoint(0)

    def on_place_button_clicked(self, checked, prompt_name):
        self.setup_prompts(skip_if_exists=True)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if checked:
            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            selectionNode.SetReferenceActivePlaceNodeClassName(self.prompt_types[prompt_name]["node_class"])
            selectionNode.SetActivePlaceNodeID(self.prompt_types[prompt_name]["node"].GetID())
            interactionNode.SetPlaceModePersistence(1)
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        else:
            if prompt_name == "lasso":
                self.submit_lasso_if_present()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def display_node_markup_point(self, display_node):
        """
        Handles the appearance of the point display node.
        """
        display_node.SetTextScale(0)  # Hide text labels
        display_node.SetGlyphScale(0.75)  # Make the points larger
        display_node.SetColor(0.0, 0.0, 1.0)  # Green color
        display_node.SetSelectedColor(0.0, 0.0, 1.0)
        display_node.SetActiveColor(0.0, 0.0, 1.0)
        display_node.SetOpacity(1.0)  # Fully opaque
        display_node.SetSliceProjection(False)  # Make points visible in all slice views

    def display_node_markup_bbox(self, display_node):
        """
        Handles the appearance of the BBox display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetInteractionHandleScale(1)
        display_node.SetGlyphScale(0)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)

    def display_node_markup_lasso(self, display_node):
        """
        Handles the appearance of the lasso display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetGlyphScale(1)
        display_node.SetLineThickness(0.3)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)

    ###############################################################################
    # Event handlers for prompts
    ###############################################################################

    #
    #  -- Point
    #
    def on_point_placed(self, caller, event):
        """
        Called when a point is placed in the scene. Grabs the point position
        and sends it to the server.
        """
        xyz = self.xyz_from_caller(caller)

        volume_node = self.get_volume_node()
        if volume_node:
            self.point_prompt(xyz=xyz, positive_click=self.is_positive)

    @ensure_synched
    def point_prompt(self, xyz=None, positive_click=False):
        """
        Uploads point prompt to the server.
        """
        url = f"{self.server}/add_point_interaction"

        seg_response = self.request_to_server(
            url, json={"voxel_coord": xyz[::-1], "positive_click": positive_click}
        )

        unpacked_segmentation = self.unpack_binary_segmentation(
            seg_response.content, decompress=False
        )
        debug_print("unpacked_segmentation.sum():", unpacked_segmentation.sum())
        debug_print(seg_response)
        debug_print(f"{positive_click} point prompt triggered! {xyz}")

        self.show_segmentation(unpacked_segmentation)
        # Save the result
        self.save_segmentation_nii("prompt", "point")

    #
    #  -- Bounding Box
    #
    def on_bbox_placed(self, caller, event):
        """
        Every time a control point is placed/moved for the bounding box ROI node.
        Once two corners are placed, we send the bounding box to the server.
        """
        xyz = self.xyz_from_caller(caller)

        if self.prev_caller is not None and caller.GetID() == self.prev_caller.GetID():
            roi_node = slicer.mrmlScene.GetNodeByID(caller.GetID())
            current_size = list(roi_node.GetSize())
            drawn_in_axis = np.argwhere(np.array(xyz) == self.prev_bbox_xyz).squeeze()
            current_size[drawn_in_axis] = 0
            roi_node.SetSize(current_size)

            volume_node = self.get_volume_node()
            if volume_node:
                outer_point_two = self.prev_bbox_xyz

                outer_point_one = [
                    xyz[0] * 2 - outer_point_two[0],
                    xyz[1] * 2 - outer_point_two[1],
                    xyz[2] * 2 - outer_point_two[2],
                ]

                self.bbox_prompt(
                    outer_point_one=outer_point_one,
                    outer_point_two=outer_point_two,
                    positive_click=self.is_positive,
                )

                def _next():
                    self.setup_prompts()
                    # Start placing a new box
                    self.ui.pbInteractionBBox.click()

                qt.QTimer.singleShot(0, _next)

            self.prev_caller = None
        else:
            self.prev_bbox_xyz = xyz

        self.prev_caller = caller

    @ensure_synched
    def bbox_prompt(self, outer_point_one, outer_point_two, positive_click=False):
        """
        Uploads BBox prompt to the server.
        """
        url = f"{self.server}/add_bbox_interaction"

        seg_response = self.request_to_server(
            url,
            json={
                "outer_point_one": outer_point_one[::-1],
                "outer_point_two": outer_point_two[::-1],
                "positive_click": positive_click,
            },
        )

        unpacked_segmentation = self.unpack_binary_segmentation(
            seg_response.content, decompress=False
        )
        self.show_segmentation(unpacked_segmentation)
        # Save the result
        self.save_segmentation_nii("prompt", "bbox")

    #
    #  -- Lasso
    #
    def on_lasso_placed(self, caller, event):
        """
        Called whenever a new point is added to the lasso.
        """
        pointsDefined = self.prompt_types["lasso"]["node"].GetNumberOfControlPoints() > 0
        self.ui.pbInteractionLassoCancel.setVisible(pointsDefined)

    def on_lasso_cancel_clicked(self):
        """
        Called when the user clicks the cancel button for the lasso.
        """
        self.prompt_types["lasso"]["node"].RemoveAllControlPoints()
        self.ui.pbInteractionLassoCancel.setVisible(False)

    def submit_lasso_if_present(self):
        """
        Submits the currently open lasso. We gather all the control points,
        rasterize them into a mask, and send the mask to the server.
        """
        caller = self.prompt_types["lasso"]["node"]
        xyzs = self.xyz_from_caller(caller, point_type="curve_point")

        if len(xyzs) < 3:
            return

        mask = self.lasso_points_to_mask(xyzs)

        volume_node = self.get_volume_node()
        if volume_node:
            self.lasso_or_scribble_prompt(
                mask=mask, positive_click=self.is_positive, tp="lasso"
            )

            def _next():
                self.setup_prompts()
                # Start placing a new lasso
                self.ui.pbInteractionLasso.click()

            qt.QTimer.singleShot(0, _next)

    #
    #  -- Scribble
    #
    def on_scribble_clicked(self, checked=False):
        """
        Activates/deactivates the hidden Segment Editor's Paint effect on the
        scribble segment (bg or fg, depending on prompt type).
        """
        self.setup_prompts(skip_if_exists=True)

        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

        if not checked:
            # Deactivate paint effect
            if self.scribble_editor_widget:
                self.scribble_editor_widget.setActiveEffectByName(
                    ""
                )  # Clears the active effect

            # Optionally clear or reset the segmentation node
            if hasattr(self, "_scribble_labelmap_callback_tag"):
                tag = self._scribble_labelmap_callback_tag.get("tag", None)
                if tag:
                    self.scribble_segment_node.RemoveObserver(tag)
                del self._scribble_labelmap_callback_tag

            return

        segment_id = "fg" if self.is_positive else "bg"

        # Set segmentation and segment
        self.scribble_editor_widget.setSegmentationNode(self.scribble_segment_node)
        self.scribble_editor_node.SetSelectedSegmentID(segment_id)

        # Set reference volume
        volume_node = self.get_volume_node()
        self.scribble_editor_widget.setSourceVolumeNode(volume_node)

        # Activate paint effect
        self.scribble_editor_widget.setActiveEffectByName("Paint")
        self.scribble_editor_widget.updateWidgetFromMRML()

        paint_effect = self.scribble_editor_widget.activeEffect()
        if paint_effect:
            paint_effect.setParameter("BrushUseAbsoluteSize", "0")  # Use relative mode
            paint_effect.setParameter("BrushSphere", "0")  # 2D brush
            paint_effect.setParameter("BrushRelativeDiameter", ".75")
            self._scribble_labelmap_callback_tag = {
                "tag": self.scribble_segment_node.AddObserver(
                    vtk.vtkCommand.AnyEvent, self.on_scribble_finished
                ),
                "label_name": segment_id,
            }
        debug_print(f"Scribble mode (hidden editor) activated on '{segment_id}'")

    #
    #  -- Lasso/scribble
    #
    @ensure_synched
    def lasso_or_scribble_prompt(self, mask, positive_click=False, tp="lasso"):
        """
        Uploads lasso or scribble prompt to the server.
        """
        if np.sum(mask) == 0:
            return
        
        url = f"{self.server}/add_{tp}_interaction"
        try:
            buffer = io.BytesIO()
            np.save(buffer, mask)
            compressed_data = gzip.compress(buffer.getvalue())

            from requests_toolbelt import MultipartEncoder

            fields = {
                "file": ("volume.npy.gz", compressed_data, "application/octet-stream"),
                "positive_click": str(
                    positive_click
                ),  # Make sure to send it as a string.
            }
            encoder = MultipartEncoder(fields=fields)
            seg_response = self.request_to_server(
                url,
                data=encoder,
                headers={
                    "Content-Type": encoder.content_type,
                    "Content-Encoding": "gzip",
                },
            )

            if seg_response.status_code == 200:
                unpacked_segmentation = self.unpack_binary_segmentation(
                    seg_response.content, decompress=False
                )
                self.show_segmentation(unpacked_segmentation)
                # Save the result
                self.save_segmentation_nii("prompt", tp)

            else:
                debug_print(
                    f"lasso_or_scribble_prompt upload failed with status code: {seg_response.status_code}"
                )
        except Exception as e:
            debug_print(f"Error in lasso_or_scribble_prompt: {e}")

    def on_scribble_finished(self, caller, event):
        """
        Called when the user completes a scribble stroke in the Paint effect.
        We calculate the diff in the drawn region and send it to the server.
        """
        debug_print("Scribble stroke finished - labelmap modified!")

        # Clean up observer if you only want it once
        if hasattr(self, "_scribble_labelmap_callback_tag"):
            caller.RemoveObserver(self._scribble_labelmap_callback_tag["tag"])
            label_name = self._scribble_labelmap_callback_tag["label_name"]
            del self._scribble_labelmap_callback_tag
        else:
            return

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.scribble_segment_node, label_name, self.get_volume_node()
        )

        if (
            hasattr(self, "_prev_scribble_mask")
            and self._prev_scribble_mask is not None
        ):
            prev_scribble_mask = self._prev_scribble_mask
        else:
            prev_scribble_mask = mask * 0

        diff_mask = mask - prev_scribble_mask
        self._prev_scribble_mask = mask

        self.lasso_or_scribble_prompt(
            mask=diff_mask, positive_click=self.is_positive, tp="scribble"
        )

        self.ui.pbInteractionScribble.click()  # turn it off
        self.ui.pbInteractionScribble.click()  # turn it on
    
    ###############################################################################
    # User input-info related functions 
    ###############################################################################
    
    def get_path_patientID_scan(self):
        import os
        # Get all scalar volume nodes
        volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        dir_path, patient_ID, exp_id = '', '', ''
        # Get path from nodes
        if volume_nodes:
            first_node = volume_nodes[0]  # Get the first node
            storage_node = first_node.GetStorageNode()
            if storage_node:
                file_path = storage_node.GetFileName()
                patient_ID = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
                exp_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                print("Patient ID:", patient_ID)
                print("Session ID", exp_id)

                dir_path = os.path.dirname(file_path)
                
            else:
                print("No storage node.")
        else:
            print("No volume nodes found!")
        return dir_path, patient_ID, exp_id
    
    def loadScans(self):
        """
        load chosen scan directory
        """

        # Choose dir of scans
        self.directory = qt.QFileDialog.getExistingDirectory()
        
        if not self.directory:
            return
        
        # Set base dir
        try:
            # for ACQUSITIONS file
            self.base_directory = os.path.dirname(os.path.dirname(self.directory))
        except Exception:
            print("No base directory defined, please check if choose the session folder!")
            pass

        # Capture AI seg checkbox state before clearing — the checkbox won't
        # fire its toggled signal if the state doesn't change, so we retrigger
        # it manually after loading the new subject.
        was_showing_seg = self.ui.ShowSegCheckBox.isChecked()
        if was_showing_seg:
            self.ui.ShowSegCheckBox.blockSignals(True)
            self.ui.ShowSegCheckBox.setChecked(False)
            self.ui.ShowSegCheckBox.blockSignals(False)

        # Store load timestamp but don't save yet
        self.pending_load_entry = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'filename': None,
            'action': "load",
            'prompt_type': None,
            'is_reset': False,
            'is_final': False
        }
        
        # Clear existing data
        self.clearLoadedData()

        # Disconnect the editor widget from the scene while loading volumes so
        # that the widget doesn't try to auto-set its source volume before a
        # segmentation node exists (which produces VTK warnings).
        self.ui.editor_widget.setMRMLScene(None)

        # Get available sessions (search recursively into subdirectories)
        sessions = sorted([
            p
            for p in Path(self.directory).rglob('*')
            if p.name.endswith('.nii.gz') or p.name.endswith('.nii')
        ])

        # Load volume and segmentation files
        for session in sessions:
            session_str = str(session)
            folder_name = session.parent.name
            # Skip everything inside segmentation_history — segs are loaded separately
            if "segmentation_history" in session_str:
                continue
            elif 'Localizer' in session_str or 'DYN' in session_str:
                continue
            else:
                node = slicer.util.loadVolume(session_str)
                if node:
                    node.SetName(folder_name)

        # Reconnect the editor widget to the scene now that volumes are loaded
        self.ui.editor_widget.setMRMLScene(slicer.mrmlScene)
        
        # Check if we're in reviewer mode (has existing segmentation history with real operations)
        history_dir = os.path.join(self.directory, "segmentation_history")
        history_path = os.path.join(history_dir, "history.json")
        has_real_operations = False
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    existing_history = json.load(f)
                has_real_operations = any(e.get('action') != 'load' for e in existing_history)
            except Exception:
                pass

        if has_real_operations:
            # Create a timestamped review session directory
            review_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.review_session_dir = os.path.join(self.directory, "review", review_ts)
            os.makedirs(self.review_session_dir, exist_ok=True)

            # Record the load event in the review session history
            load_entry = {
                'timestamp': review_ts,
                'action': "review_load",
                'notes': "Opened folder for review"
            }
            self.save_review_history(load_entry)

            # Load all segmentations from segmentation_history/segs/
            segs_dir = os.path.join(history_dir, "segs")
            loaded_seg_names = set()
            self.registered_seg_nodes = []
            if os.path.isdir(segs_dir):
                for fname in sorted(os.listdir(segs_dir)):
                    if not (fname.endswith('.nii.gz') or fname.endswith('.nii')):
                        continue
                    if fname in loaded_seg_names:
                        continue
                    seg_path = os.path.join(segs_dir, fname)
                    seg_node = slicer.util.loadSegmentation(seg_path)
                    if seg_node:
                        loaded_seg_names.add(fname)
                        # Name without _seg.nii.gz suffix → matches the volume name
                        vol_name = fname.replace('_seg.nii.gz', '').replace('_seg.nii', '')
                        seg_node.SetName(f"{vol_name}_seg")
                        ref_vol = slicer.mrmlScene.GetFirstNodeByName(vol_name)
                        if ref_vol:
                            seg_node.SetReferenceImageGeometryParameterFromVolumeNode(ref_vol)
                        # Hide by default; reviewer can isolate/toggle individually
                        dn = seg_node.GetDisplayNode()
                        if dn:
                            dn.SetVisibility(False)
                        # Track pairs so on_final_save can save each seg to the right volume
                        if ref_vol:
                            self.registered_seg_nodes.append((seg_node, ref_vol))

            self.ui.ReviewPanel.setVisible(True)
            self.ui.DiagnosisBox.setVisible(True)
            self.ui.groupBox_3.setVisible(True)
            slicer.util.infoDisplay(
                f"Loaded {len(loaded_seg_names)} segmentation(s) from history for review.",
                windowTitle="Review Mode"
            )
        else:
            # First-time segmentation (no history, or history only contains load entries)
            self.review_session_dir = None
            self.ui.ReviewPanel.setVisible(False)
            self.ui.DiagnosisBox.setVisible(False)
            self.ui.groupBox_3.setVisible(False)
            self.segmentation_history = [self.pending_load_entry]
            self.save_history_file()
            self.pending_load_entry = None
        
        # Re-initialize prompts with the new volume (scribble node needs new geometry)
        self.setup_prompts()

        self.updateInfo()

        # Retrigger AI seg if it was showing before load
        if was_showing_seg:
            self.ui.ShowSegCheckBox.setChecked(True)  # fires onShowSegToggled → updateAISegmentation

    def updateInfo(self):

        # Get pid and scan info
        scan_dir, patient_ID, _ = self.get_path_patientID_scan()

        # # Clinical info (requires CSV on local machine)
        # import pandas as pd
        # scan_dir, patient_ID, exp_id = self.get_path_patientID_scan()
        # data = pd.read_csv('/home/xwan/Documents/Osteosarcoma/os_data_tmp/image_records/Osteo_Sarcoma_xnatsort_20250319_0707_local_paths_mapped_labels.csv')
        # if patient_ID != '':
        #     loc = data[(data['Subject'] == patient_ID) & (data['Experiment'] == exp_id)].loc_prim_code.values[0]
        #     baseline_info = data[(data['Subject'] == patient_ID) & (data['Experiment'] == exp_id)].Before_after_NAC.values[0]
        #     self.ui.LocationLabel.text = f'{loc}'
        #     self.ui.LocationLabel.styleSheet = "color: green" if self.ui.LocationLabel.text != 'None' else "color: Black"
        #     self.ui.BaselineLabel.text = f'{baseline_info}'

        if patient_ID != '':
            self.ui.PID.text = f'{patient_ID}'
            self.ui.PID.styleSheet = "color: green" if self.ui.PID.text != 'None' else "color: Black"
        else:
            print('No image found.')
    
    def onPatientInfoToggled(self, checked):
        """Show/hide ClinicalInfoLabel based on PatientInfoBox checkbox state."""
        # Only show if we actually have info loaded (non-empty label text)
        has_info = bool(self.ui.ClinicalInfoLabel.text.strip())
        self.ui.ClinicalInfoLabel.setVisible(checked and has_info)

    def _get_assessment_json_path(self):
        """Return path to the assessment JSON in the active review session directory."""
        if not self.directory:
            return None
        folder = self.review_session_dir if self.review_session_dir else self.directory
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "assessment.json")

    def _load_assessment_json(self, path):
        """Load existing assessment JSON or return empty dict."""
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_assessment_json(self, path, data):
        """Save assessment data to JSON file."""
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            debug_print(f"Error saving assessment JSON: {e}")
            return False

    def on_submit_anatomy(self):
        """Save tumor location from combo box or custom text edit to assessment JSON."""
        if not self.directory:
            slicer.util.warningDisplay("Please load scans first.", windowTitle="No Directory")
            return

        custom_text = self.ui.customAnatomyEdit.text.strip()
        if custom_text:
            location = custom_text
        else:
            location = self.ui.AnatomyBox.currentText.strip()

        if not location or location == "Loading...":
            slicer.util.warningDisplay("Please select or enter an anatomy location.", windowTitle="No Anatomy")
            return

        path = self._get_assessment_json_path()
        data = self._load_assessment_json(path)
        data['tumor_location'] = location
        data['tumor_location_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self._save_assessment_json(path, data):
            slicer.util.infoDisplay(f"Anatomy location saved: {location}", windowTitle="Saved")
        else:
            slicer.util.errorDisplay("Failed to save anatomy location.", windowTitle="Save Error")

    def on_submit_diagnosis(self):
        """Save tumor type and confidence level to assessment JSON."""
        if not self.directory:
            slicer.util.warningDisplay("Please load scans first.", windowTitle="No Directory")
            return

        tumor_type = self.ui.comboBox_2.currentText.strip()
        confidence = self.ui.comboBox.currentText.strip()

        path = self._get_assessment_json_path()
        data = self._load_assessment_json(path)
        data['diagnosis_tumor_type'] = tumor_type
        data['diagnosis_confidence'] = confidence
        data['diagnosis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self._save_assessment_json(path, data):
            slicer.util.infoDisplay(
                f"Diagnosis saved:\n  Type: {tumor_type}\n  Confidence: {confidence}",
                windowTitle="Saved"
            )
        else:
            slicer.util.errorDisplay("Failed to save diagnosis.", windowTitle="Save Error")

    def clearLoadedData(self):
        """Remove all volumes and segmentations from the scene"""
        # Detach editor widgets before removing nodes to prevent Slicer from
        # auto-creating undo checkpoint files when nodes disappear unexpectedly.
        self.ui.editor_widget.setSegmentationNode(None)
        self.ui.editor_widget.setSourceVolumeNode(None)

        # Clear the AI seg reference so it isn't double-removed
        self.ai_seg_node = None

        # Reset summary label and anatomy box for the new subject
        self.ui.segSummaryLabel.setText("")
        self.ui.AnatomyBox.clear()

        # Remove volumes
        volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        for node in volume_nodes:
            slicer.mrmlScene.RemoveNode(node)

        # Remove segmentations
        seg_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        for node in seg_nodes:
            slicer.mrmlScene.RemoveNode(node)
        print("Cleared all previously loaded data")
            
    ###############################################################################
    # AI bone segmentation display functions
    ###############################################################################

    def onShowSegToggled(self, checked):
        """Called when the Show AI Bone Seg checkbox is toggled."""
        if checked:
            self.updateAISegmentation()
            self.ui.segSummaryLabel.setVisible(True)
        else:
            self.hideAISegmentation()
            self.ui.segSummaryLabel.setVisible(False)

    def getAISegPath(self):
        """
        Returns (seg_file_path, labels_file_path) for the current volume, or (None, None).
        Expects the same relative path under self.seg_directory as under self.directory.
        Also updates segSummaryLabel with the top-3 anatomy labels across all sessions
        for the current subject.
        """
        if not self.seg_directory or not self.directory:
            return None, None

        volume_node = self.get_volume_node()
        if not volume_node:
            return None, None

        storage_node = volume_node.GetStorageNode()
        if not storage_node:
            return None, None

        image_path = Path(storage_node.GetFileName())
        try:
            rel_path = image_path.parent.relative_to(self.base_directory)
        except ValueError:
            return None, None

        seg_dir = Path(self.seg_directory) / "sorted_data" /rel_path
        print("Seg dir path:", self.seg_directory)
        print("Rel path:", rel_path)
        print("Img path", image_path)
        
        # seg_file = seg_dir / "segmentations.nii.gz"
        seg_file = seg_dir / "segmentation.nii"
        labels_file = seg_dir / "bone_seg_labels.json"

        print("Segmentation is from dir:", seg_dir)
        print("Segmentation is from path:", seg_file)
        # Build subject-level summary (parent of session = patient/study dir)
        subject_seg_dir = seg_dir.parent
        self._updateSegSummaryLabel(subject_seg_dir)

        if not seg_file.exists():
            print("test")
            return None, None

        return str(seg_file), str(labels_file) if labels_file.exists() else None

    def _updateSegSummaryLabel(self, subject_seg_dir):
        """
        Scans all bone_seg_labels.json files under subject_seg_dir,
        counts every anatomy label, and displays the top 3 in segSummaryLabel.
        """
        import json
        from collections import Counter

        label_counts = Counter()
        total_files = 0
        for json_file in subject_seg_dir.glob("*/bone_seg_labels.json"):
            try:
                with open(json_file) as f:
                    labels = json.load(f)
                total_files += 1
                for v in labels.values():
                    if isinstance(v, str):
                        label_counts[v] += 1
            except Exception:
                pass

        self.ui.AnatomyBox.clear()
        if label_counts:
            top = label_counts.most_common(3)
            parts = [f"{name} ({count}/{total_files})" for name, count in top]
            summary = "Top structures: " + "; ".join(parts)
            for name, _ in top:
                self.ui.AnatomyBox.addItem(name)
        else:
            summary = ""
        self.ui.segSummaryLabel.setText(summary)

    def updateAISegmentation(self):
        """Load and display the AI bone segmentation for the current volume."""
        self.hideAISegmentation()

        if not self.seg_directory:
            slicer.util.warningDisplay(
                "AI segmentation directory is not set.",
                windowTitle="No Segmentation Directory",
            )
            self.ui.ShowSegCheckBox.setChecked(False)
            return

        seg_path, labels_path = self.getAISegPath()

        if seg_path is None:
            slicer.util.warningDisplay(
                "No AI segmentation found for this image.",
                windowTitle="No Segmentation Found",
            )
            self.ui.ShowSegCheckBox.setChecked(False)
            return

        node = slicer.util.loadSegmentation(seg_path)
        if node:
            volume_node = self.get_volume_node()
            node.SetName(f"Total_Seg_{volume_node.GetName() if volume_node else 'unknown'}")

            if labels_path:
                import json
                with open(labels_path) as f:
                    labels = json.load(f)
                seg = node.GetSegmentation()
                for i in range(seg.GetNumberOfSegments()):
                    seg_id = seg.GetNthSegmentID(i)
                    segment = seg.GetSegment(seg_id)
                    # Try label index starting from 1, then 0
                    name = labels.get(str(i + 1)) or labels.get(str(i))
                    if name:
                        segment.SetName(str(name))

            self.ai_seg_node = node
            self._last_volume_id = volume_node.GetID() if volume_node else None

    def hideAISegmentation(self):
        """Remove the AI segmentation node from the scene."""
        if self.ai_seg_node and slicer.mrmlScene.IsNodePresent(self.ai_seg_node):
            slicer.mrmlScene.RemoveNode(self.ai_seg_node)
        self.ai_seg_node = None

    def on_active_volume_changed(self, caller, event):
        """Called when the active volume in the scene changes."""
        if not self.ui.ShowSegCheckBox.isChecked():
            return
        current_volume = self.get_volume_node()
        current_id = current_volume.GetID() if current_volume else None
        if current_id != self._last_volume_id:
            # Debounce: wait briefly in case multiple nodes change rapidly (e.g., during load)
            qt.QTimer.singleShot(300, self._debounced_update_ai_seg)

    def _debounced_update_ai_seg(self):
        """Deferred update — only fires if the volume is still different from last shown."""
        if not self.ui.ShowSegCheckBox.isChecked():
            return
        current_volume = self.get_volume_node()
        current_id = current_volume.GetID() if current_volume else None
        if current_id != self._last_volume_id:
            self.updateAISegmentation()

    def saveResults(self):
        import os
        import datetime
        import json
        
        try:
            scan_dir, _, _ = self.get_path_patientID_scan()
            if not scan_dir or not os.path.exists(scan_dir):
                raise ValueError("Scan directory does not exist or is invalid")
                
            outputFile = os.path.join(scan_dir, 'bone_seg_notes.json')
            
            res = {
                'status': 'completed',
                'timestamp': '',
                'notes': ''
            }

            # Get values from UI
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notes = self.ui.plainTextEdit.toPlainText()

            # Update result dictionary
            res.update({
                'timestamp': timestamp,
                'notes': notes
            })
            
            # Write to file with pretty formatting
            with open(outputFile, 'w') as outfile:
                json.dump(res, outfile, indent=4)
                
            # Optional: Show success message in Slicer
            slicer.util.infoDisplay(f"Results saved to {outputFile}", windowTitle="Save Successful")
            
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to save results: {str(e)}", windowTitle="Save Error")
            import traceback
            traceback.print_exc()
    
    ###############################################################################
    # Segmentation-related functions
    ###############################################################################

    def make_new_segment(self):
        """
        Creates a new empty segment in the current segmentation, increments a name,
        and sets it as the selected segment.
        """
        # After creating a new segment, negative prompts do not make sense, so
        # we're automatically switching the prompt type to positive.
        self.ui.pbPromptTypePositive.click()
        
        debug_print("doing make_new_segment")
        segmentation_node = self.get_segmentation_node()

        # Generate a new segment name
        segment_ids = segmentation_node.GetSegmentation().GetSegmentIDs()
        if len(segment_ids) == 0:
            new_segment_name = "Segment_1"
        else:
            # Find the next available number
            segment_numbers = [
                int(seg.split("_")[-1])
                for seg in segment_ids
                if seg.startswith("Segment_") and seg.split("_")[-1].isdigit()
            ]
            next_segment_number = max(segment_numbers) + 1 if segment_numbers else 1
            new_segment_name = f"Segment_{next_segment_number}"

        # Create and add the new segment
        new_segment_id = segmentation_node.GetSegmentation().AddEmptySegment(
            new_segment_name
        )
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)

        # Make sure the right node is selected
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)

        return segmentation_node, new_segment_id

    def clear_current_segment(self):
        """
        Clears the contents (labelmap) of the currently selected segment
        and updates the server.
        """
        # After clearing a segment, negative prompts do not make sense, so
        # we're automatically switching the prompt type to positive.
        self.ui.pbPromptTypePositive.click()
        
        _, selected_segment_id = self.get_selected_segmentation_node_and_segment_id()

        if selected_segment_id:
            debug_print(f"Clearing segment: {selected_segment_id}")

            # Save empty segmentation before clearing
            reset_path = self.save_segmentation_nii("reset")
    
            self.show_segmentation(
                np.zeros(self.get_image_data().shape, dtype=np.uint8)
            )
            self.setup_prompts()
            self.upload_segment_to_server()
        else:
            debug_print("No segment selected to clear.")

    def show_segmentation(self, segmentation_mask):
        """
        Updates the currently selected segment with the given binary mask array.
        """
        t0 = time.time()
        self.previous_states["segment_data"] = segmentation_mask

        segmentationNode, selectedSegmentID = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        was_3d_shown = segmentationNode.GetSegmentation().ContainsRepresentation(slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName())

        with slicer.util.RenderBlocker():  # avoid flashing of 3D view
            self.ui.editor_widget.saveStateForUndo()
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                segmentation_mask,
                segmentationNode,
                selectedSegmentID,
                self.get_volume_node(),
            )
            if was_3d_shown:
                segmentationNode.CreateClosedSurfaceRepresentation()

        # Mark the segment as being edited (can be useful for selective saving of only modified segments)
        segment = segmentationNode.GetSegmentation().GetSegment(selectedSegmentID)
        if slicer.vtkSlicerSegmentationsModuleLogic.GetSegmentStatus(segment) == slicer.vtkSlicerSegmentationsModuleLogic.NotStarted:
            slicer.vtkSlicerSegmentationsModuleLogic.SetSegmentStatus(segment, slicer.vtkSlicerSegmentationsModuleLogic.InProgress)

        # Mark the segmentation as modified so the UI updates
        segmentationNode.Modified()

        if segmentation_mask.sum() > 0:
            # If we do this when segmentation_mask.sum() == 0, sometimes Slicer will throw "bogus" OOM errors
            # (see https://github.com/coendevente/SlicerNNInteractive/issues/38)
            segmentationNode.GetSegmentation().CollapseBinaryLabelmaps()
        
        del segmentation_mask

        debug_print(f"show_segmentation took {time.time() - t0}")

    def get_segmentation_node(self):
        """
        Returns the currently referenced segmentation node (from the Segment Editor).
        If none exists, we create a fresh one.
        """
        # If the segmentation widget has a currently selected segmentation node, return it.
        segmentation_node = self.ui.editor_widget.segmentationNode()
        if segmentation_node:
            if segmentation_node.GetName() != self.scribble_segment_node_name:
                return segmentation_node

        # Otherwise, fall back to getting the first suitable segmentation node
        segmentation_node = None
        segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        for segmentation_node in segmentation_nodes:
            if segmentation_node.GetName() == self.scribble_segment_node_name:
                segmentation_node = None
                continue

        # Create new segmentation node if none suitable found
        if not segmentation_node:
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        # Set segmentation node in widget
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.get_volume_node())

        return segmentation_node

    def get_selected_segmentation_node_and_segment_id(self):
        """
        Retrieve the currently selected segmentation node & segment ID.
        If none, create one.
        """
        debug_print("doing get_selected_segmentation_node_and_segment_id")
        segmentation_node = self.get_segmentation_node()
        selected_segment_id = self.get_current_segment_id()
        if not selected_segment_id:
            return self.make_new_segment()

        return segmentation_node, selected_segment_id

    def get_current_segment_id(self):
        """
        Returns the ID of the segment currently selected in the segment editor.
        """
        return self.ui.editor_widget.mrmlSegmentEditorNode().GetSelectedSegmentID()

    def get_segment_data(self):
        """
        Gets the labelmap array (binary) of the currently selected segment.
        """
        segmentation_node, selected_segment_id = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, selected_segment_id, self.get_volume_node()
        )
        seg_data_bool = mask.astype(bool)

        return seg_data_bool

    def selected_segment_changed(self):
        """
        Checks if the current segment mask has changed from our `self.previous_states`.
        """
        segment_data = self.get_segment_data()
        old_segment_data = self.previous_states.get("segment_data", None)
        selected_segment_changed = old_segment_data is None or not np.array_equal(
            old_segment_data.astype(bool), segment_data.astype(bool)
        )

        debug_print(f"segment_data.sum(): {segment_data.sum()}")

        if old_segment_data is not None:
            debug_print(f"old_segment_data.sum(): {old_segment_data.sum()}")
        else:
            debug_print("old_segment_data is None")

        debug_print(f"selected_segment_changed: {selected_segment_changed}")

        return selected_segment_changed

    ###############################################################################
    # Server communication and sync functions
    ###############################################################################

    def update_server(self):
        """
        Reads user-entered server URL from UI, saves to QSettings, updates self.server.
        """
        self.server = self.ui.Server.text.rstrip("/")
        settings = qt.QSettings()
        settings.setValue("SlicerNNInteractive/server", self.server)
        debug_print(f"Server URL updated and saved: {self.server}")

    def request_to_server(self, *args, **kwargs):
        """
        Wraps requests.post in a try/except and shows error in pop up windows if necessary.
        """

        with slicer.util.tryWithErrorDisplay(_("Segmentation failed."), waitCursor=True):

            error_message = None
            try:
                response = requests.post(*args, **kwargs)
                debug_print('response:', response)
            except requests.exceptions.MissingSchema as e:
                response = None
                if self.server == "":
                    raise RuntimeError("It seems you have not set the server URL yet. You can configure it in the 'Configuration' tab.")
                else:
                    raise RuntimeError(f"Server URL '{self.server}' is unreachable. You can edit the URL in the 'Configuration' tab.")
            except requests.exceptions.ConnectionError as e:
                response = None
                raise RuntimeError(f"Failed to connect to server '{self.server}'. Please make sure the server is running and check the server URL in the 'Configuration' tab.")
            except requests.exceptions.InvalidSchema as e:
                append_text_to_error_message = ""
                if not args[0].startswith("http://"):
                    append_text_to_error_message = "\n\nHint: Perhaps your Server URL in the 'Configuration' tab should start with 'http://'. For example, if your server runs on localhost and port 1527, 'localhost:1527' would not work as a Server URL, while 'http://localhost:1527' would."
                raise RuntimeError(f'{e}{append_text_to_error_message}')

            if response.status_code != 200:
                status_code = response.status_code
                response = None
                raise RuntimeError(f"Something has gone wrong with your request (Status code {status_code}).")

            t0 = time.time()
            # Try to parse JSON and check for a specific error.
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                resp_json = response.json()
                if resp_json.get("status") == "error":
                    if "No image uploaded" in resp_json.get("message", ""):
                        debug_print("No image has been uploaded to the server. Please upload an image first.")
                        self.upload_image_to_server()
                        self.upload_segment_to_server()
                        return self.request_to_server(*args, **kwargs)
                    else:
                        response = None
                        raise RuntimeError(f"Server error: {resp_json.get('message', 'Unknown error')}")

            debug_print('1157 took', time.time() - t0)

        return response

    def upload_image_to_server(self):
        """
        Gets volume data from Slicer, packs it, and uploads it to the server.
        """
        debug_print("Syncing image with server...")
        try:
            # Retrieve image data, window, and level.
            t0 = time.time()
            image_data = (
                self.get_image_data()
            )  # Expected to return (image_data, window, level)
            debug_print(f"self.get_image_data took {time.time() - t0}")

            if image_data is None:
                debug_print("No image data available to upload.")
                return

            t0 = time.time()
            url = (
                f"{self.server}/upload_image"  # Update this with your actual endpoint.
            )

            buffer = io.BytesIO()
            np.save(buffer, image_data)
            raw_data = buffer.getvalue()
            debug_print(f"len(raw_data): {len(raw_data)}")

            files = {"file": ("volume.npy", raw_data, "application/octet-stream")}

            # Create your MultipartEncoder without gzip headers
            from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

            slicer.progress_window = slicer.util.createProgressDialog(autoClose=False)
            slicer.progress_window.minimum = 0
            slicer.progress_window.maximum = 100
            slicer.progress_window.setLabelText("Uploading image...")

            def my_callback(monitor):
                if not hasattr(monitor, "last_update"):
                    monitor.last_update = time.time()
                if time.time() - monitor.last_update <= 0.2:
                    return
                monitor.last_update = time.time()
                slicer.progress_window.setValue(
                    monitor.bytes_read / len(raw_data) * 100
                )
                slicer.progress_window.show()
                slicer.progress_window.activateWindow()
                slicer.progress_window.setLabelText("Uploading image...")
                slicer.app.processEvents()

            encoder = MultipartEncoder(fields=files)
            monitor = MultipartEncoderMonitor(encoder, my_callback)

            try:
                result = self.request_to_server(
                    url, data=monitor, headers={"Content-Type": monitor.content_type}
                )
            finally:
                slicer.progress_window.close()

            return result
        except Exception as e:
            debug_print(f"Error in upload_image_to_server: {e}")

    def upload_segment_to_server(self):
        """
        Grabs current segmentation labelmap, gzips it, and sends it to the server.
        """
        debug_print("Syncing segment with server...")
        try:
            segment_data = self.get_segment_data()
            files = self.mask_to_np_upload_file(segment_data)
            url = f"{self.server}/upload_segment"  # Update this with your actual endpoint.

            result = self.request_to_server(
                url, files=files, headers={"Content-Encoding": "gzip"}
            )

            return result
        except Exception as e:
            debug_print(f"Error in upload_image_to_server: {e}")

    ###############################################################################
    # Utility / converters functions
    ###############################################################################

    def get_image_data(self):
        """
        Returns the voxel data of the current active (or first available) volume node.
        """
        volume_node = self.get_volume_node()
        if volume_node:
            return slicer.util.arrayFromVolume(volume_node)

        return None

    def get_volume_node(self):
        """
        Retrieves the current source volume node chosen in the segment editor widget.
        If nothing is set then use the most recently added scalar volume
        """
        # Get volume node from segment editor widget
        volumeNode = self.ui.editor_widget.sourceVolumeNode()

        if not volumeNode:
            # Get the most recently added volume node
            volumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if volumeNodes:
                volumeNode = volumeNodes[-1]
            # Only push to the editor widget if a segmentation node is already
            # set — otherwise Slicer emits "need to set segment editor and
            # segmentation nodes first" VTK warnings.
            if volumeNode and self.ui.editor_widget.segmentationNode():
                self.ui.editor_widget.setSourceVolumeNode(volumeNode)

        return volumeNode

    def image_changed(self, do_prev_image_update=True):
        """
        Checks if the volume's voxel data changed since the last time we stored it.
        """
        image_data = self.get_image_data()
        if image_data is None:
            debug_print("No volume node found")
            return

        old_image_data = self.previous_states.get("image_data", None)

        image_changed = old_image_data is None or not np.array_equal(
            old_image_data, image_data
        )

        if do_prev_image_update:
            self.previous_states["image_data"] = copy.deepcopy(image_data)

        return image_changed

    def mask_to_np_upload_file(self, mask):
        """
        Converts a numpy mask into a gzipped file object for POSTing.
        """
        buffer = io.BytesIO()
        np.save(buffer, mask)
        compressed_data = gzip.compress(buffer.getvalue())

        files = {"file": ("volume.npy.gz", compressed_data, "application/octet-stream")}

        return files

    def unpack_binary_segmentation(self, binary_data, decompress=False):
        """
        Unpacks data received from server into a full 3D numpy array (bool).
        """
        if decompress:
            binary_data = binary_data = gzip.decompress(binary_data)

        if self.get_image_data() is None:
            self.capture_image()

        vol_shape = self.get_image_data().shape
        total_voxels = np.prod(vol_shape)
        unpacked_bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
        unpacked_bits = unpacked_bits[:total_voxels]

        segmentation_mask = (
            unpacked_bits.reshape(vol_shape).astype(np.bool_).astype(np.uint8)
        )

        return segmentation_mask

    def ras_to_xyz(self, pos):
        """
        Converts an RAS position to IJK voxel coords in the current volume node.
        """
        volumeNode = self.get_volume_node()

        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas
        )
        point_VolumeRas = transformRasToVolumeRas.TransformPoint(pos)

        volumeRasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
        point_Ijk = [0, 0, 0, 1]
        volumeRasToIjk.MultiplyPoint(list(point_VolumeRas) + [1.0], point_Ijk)
        xyz = [int(round(c)) for c in point_Ijk[0:3]]
        return xyz


    def xyz_from_caller(self, caller, lock_point=True, point_type="control_point"):
        """
        Extract voxel coordinates from a Markups node.
        `point_type` can be either "control_point" or "curve_point".
        """
        if point_type == "control_point":
            n = caller.GetNumberOfControlPoints()
            if n < 0:
                debug_print("No control points found")
                return

            pos = [0, 0, 0]
            caller.GetNthControlPointPosition(n - 1, pos)
            if lock_point:
                caller.SetNthControlPointLocked(n - 1, True)
            xyz = self.ras_to_xyz(pos)
            return xyz
        elif point_type == "curve_point":
            vtk_pts = caller.GetCurvePointsWorld()
            
            if vtk_pts is not None:
                vtk_pts_data = vtk_to_numpy(vtk_pts.GetData())
                xyz = [self.ras_to_xyz(pos) for pos in vtk_pts_data]
                debug_print(xyz)
                return xyz

            return []
        else:
            raise ValueError(f'Unknown point_type {point_type}')

    def lasso_points_to_mask(self, points):
        """
        Given a list of voxel coords (defining a polygon in one slice),
        returns a 3D mask with that polygon filled in the appropriate slice.
        """
        from skimage.draw import polygon

        shape = self.get_image_data().shape
        pts = np.array(points)  # shape (n, 3)

        # Determine which coordinate is constant
        const_axes = [i for i in range(3) if np.unique(pts[:, i]).size == 1]
        if len(const_axes) != 1:
            raise ValueError(
                "Expected exactly one constant coordinate among the points"
            )
        const_axis = const_axes[0]
        const_val = int(pts[0, const_axis])

        # Create a blank 3D mask
        mask = np.zeros(shape, dtype=np.uint8)

        # Depending on which axis is constant, extract the 2D polygon and fill the corresponding slice.
        # Note: our volume is ordered as (z, y, x)
        if const_axis == 2:
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            rr, cc = polygon(y_coords, x_coords, shape=(shape[1], shape[2]))
            mask[const_val, rr, cc] = 1
        elif const_axis == 1:
            x_coords = pts[:, 0]
            z_coords = pts[:, 2]
            rr, cc = polygon(z_coords, x_coords, shape=(shape[0], shape[2]))
            mask[rr, const_val, cc] = 1
        elif const_axis == 0:
            y_coords = pts[:, 1]
            z_coords = pts[:, 2]
            rr, cc = polygon(z_coords, y_coords, shape=(shape[0], shape[1]))
            mask[rr, cc, const_val] = 1

        return mask

    ###############################################################################
    # Prompt type toggle (positive / negative)
    ###############################################################################

    @property
    def is_positive(self):
        """
        Returns True if the current prompt is set to "positive",
        False if "negative."
        """
        return self.ui.pbPromptTypePositive.isChecked()

    def on_prompt_type_positive_clicked(self, checked=False):
        """
        Called when user presses the "Positive" prompt button.
        """
        # Update UI
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypePositive.setChecked(True)
        self.ui.pbPromptTypeNegative.setChecked(False)
        debug_print("Prompt type set to POSITIVE")

    def on_prompt_type_negative_clicked(self, checked=False):
        """
        Called when user presses the "Negative" prompt button.
        """

        # Update UI
        self.current_prompt_type_positive = False
        self.ui.pbPromptTypePositive.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypePositive.setChecked(False)
        self.ui.pbPromptTypeNegative.setChecked(True)
        debug_print("Prompt type set to NEGATIVE")

    def toggle_prompt_type(self, checked=False):
        """
        Toggle between positive and negative (triggered by 'T' key).
        """
        debug_print("Toggling prompt type (positive <> negative)")
        if self.current_prompt_type_positive:
            self.on_prompt_type_negative_clicked()
        else:
            self.on_prompt_type_positive_clicked()
