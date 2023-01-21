# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import importlib
import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

# Cannot be imported lazily because they would be misinterpreted as a module by LazyImporter
from htc.settings import settings
from htc.settings_seg import settings_seg

# If you want to add new imports to this file, please add them both, to the _import_structure dict and inside the TYPE_CHECKING check

# We lazy-load all import htc functionality into the htc namespace (e.g. so that users can write from htc import DataPath)
# A direct import (like guarded in TYPE_CHECKING below) would cost a lot of time since we have many dependencies and this would require loading all of them even if not needed
# This way, the import cost is only payed when the functionality is actually needed
_import_structure = {
    "cpp": [
        "hierarchical_bootstrapping",
        "kfold_combinations",
        "map_label_image",
        "nunique",
        "segmentation_mask",
        "spxs_predictions",
        "tensor_mapping",
    ],
    "evaluation": ["evaluation"],  # Also make modules itself importable
    "evaluation.analyze_tfevents": ["read_tfevent_losses"],
    "evaluation.evaluate_images": ["calc_dice_metric", "calc_surface_dice", "calc_surface_distance", "evaluate_images"],
    "evaluation.evaluate_superpixels": ["EvaluateSuperpixelImage"],
    "evaluation.metrics.ECELoss": ["ECELoss"],
    "evaluation.metrics.scores": ["normalize_grouped_cm"],
    "fonts": ["fonts"],
    "fonts.set_font": ["set_font"],
    "model_processing": ["model_processing"],
    "model_processing.ImageConsumer": ["ImageConsumer"],
    "model_processing.Runner": ["Runner"],
    "model_processing.TestLeaveOneOutPredictor": ["TestLeaveOneOutPredictor"],
    "model_processing.TestPredictor": ["TestPredictor"],
    "model_processing.ValidationPredictor": ["ValidationPredictor"],
    "models": ["models"],
    "models.common.class_weights": ["calculate_class_weights"],
    "models.common.distance_correlation": ["distance_correlation", "distance_correlation_features"],
    "models.common.ForwardHookPromise": ["ForwardHookPromise"],
    "models.common.HierarchicalSampler": ["HierarchicalSampler"],
    "models.common.HTCDataset": ["HTCDataset"],
    "models.common.HTCDatasetStream": ["HTCDatasetStream"],
    "models.common.HTCLightning": ["HTCLightning"],
    "models.common.HTCModel": ["HTCModel"],
    "models.common.MetricAggregation": ["MetricAggregation"],
    "models.common.StreamDataLoader": ["StreamDataLoader"],
    "models.common.torch_helpers": [
        "FlexibleIdentity",
        "copy_sample",
        "cpu_only_tensor",
        "move_batch_gpu",
        "pad_tensors",
        "smooth_one_hot",
        "str_to_dtype",
    ],
    "models.common.utils": ["get_n_classes", "infer_swa_lr", "samples_equal"],
    "models.data.DataSpecification": ["DataSpecification"],
    "models.data.SpecsGeneration": ["SpecsGeneration"],
    "models.image.DatasetImage": ["DatasetImage"],
    "models.image.DatasetImageBatch": ["DatasetImageBatch"],
    "models.image.DatasetImageStream": ["DatasetImageStream"],
    "models.image.LightningImage": ["LightningImage"],
    "models.image.ModelImage": ["ModelImage"],
    "models.patch.DatasetPatchImage": ["DatasetPatchImage"],
    "models.patch.DatasetPatchStream": ["DatasetPatchStream"],
    "models.patch.LightningPatch": ["LightningPatch"],
    "models.pixel.DatasetPixelStream": ["DatasetPixelStream"],
    "models.pixel.LightningPixel": ["LightningPixel"],
    "models.pixel.ModelPixel": ["ModelPixel"],
    "models.pixel.ModelPixelRGB": ["ModelPixelRGB"],
    "models.superpixel_classification.DatasetSuperpixelImage": ["DatasetSuperpixelImage"],
    "models.superpixel_classification.DatasetSuperpixelStream": ["DatasetSuperpixelStream"],
    "models.superpixel_classification.LightningSuperpixelClassification": ["LightningSuperpixelClassification"],
    "models.superpixel_classification.ModelSuperpixelClassification": ["ModelSuperpixelClassification"],
    "tissue_atlas": ["tissue_atlas"],
    "tissue_atlas.median_pixel.DatasetMedianPixel": ["DatasetMedianPixel"],
    "tissue_atlas.median_pixel.LightningMedianPixel": ["LightningMedianPixel"],
    "tissue_atlas.MetricAggregationClassification": ["MetricAggregationClassification"],
    "tissue_atlas.settings_atlas": ["settings_atlas"],
    "tivita": ["tivita"],
    "tivita.colorscale": ["tivita_colorscale"],
    "tivita.DataPath": ["DataPath"],
    "tivita.DatasetSettings": ["DatasetSettings"],
    "tivita.hsi": ["read_tivita_dark", "read_tivita_hsi", "tivita_wavelengths"],
    "tivita.metadata": ["generate_metadata_table", "read_meta_file"],
    "tivita.rgb": ["hsi_to_rgb", "read_tivita_rgb"],
    "utils": ["utils"],
    "utils.AdvancedJSONEncoder": ["AdvancedJSONEncoder"],
    "utils.blosc_compression": ["compress_file", "decompress_file"],
    "utils.ColorcheckerReader": ["ColorcheckerReader"],
    "utils.ColoredFileLog": ["ColoredFileLog"],
    "utils.colors": ["generate_distinct_colors"],
    "utils.Config": ["Config"],
    "utils.DatasetDir": ["DatasetDir"],
    "utils.DelayedFileHandler": ["DelayedFileHandler"],
    "utils.DomainMapper": ["DomainMapper"],
    "utils.DuplicateFilter": ["DuplicateFilter"],
    "utils.general": [
        "apply_recursive",
        "clear_directory",
        "merge_dicts_deep",
        "safe_copy",
        "sha256_file",
        "subprocess_run",
    ],
    "utils.helper_functions": [
        "basic_statistics",
        "group_median_spectra",
        "median_table",
        "sort_labels",
        "sort_labels_cm",
        "utilization_table",
    ],
    "utils.LabelMapping": ["LabelMapping"],
    "utils.LDA": ["LDA"],
    "utils.MeasureTime": ["MeasureTime"],
    "utils.MultiPath": ["MultiPath"],
    "utils.parallel": ["p_imap", "p_map"],
    "utils.SpectrometerReader": ["SpectrometerReader"],
    "utils.sqldf": ["sqldf"],
    "utils.type_from_string": ["type_from_string"],
    "utils.unify_path": ["unify_path"],
    "utils.visualization": [
        "add_std_fill",
        "compress_html",
        "create_class_scores_figure",
        "create_confusion_figure",
        "create_confusion_figure_comparison",
        "create_ece_figure",
        "create_overview_document",
        "create_running_metric_plot",
        "create_segmentation_overlay",
        "create_surface_dice_plot",
        "create_training_stats_figure",
        "prediction_figure_html",
        "show_loss_chart",
        "visualize_dict",
    ],
}

if TYPE_CHECKING:
    from htc.cpp import (
        hierarchical_bootstrapping,
        kfold_combinations,
        map_label_image,
        nunique,
        segmentation_mask,
        spxs_predictions,
        tensor_mapping,
    )
    from htc.evaluation.analyze_tfevents import read_tfevent_losses
    from htc.evaluation.evaluate_images import (
        calc_dice_metric,
        calc_surface_dice,
        calc_surface_distance,
        evaluate_images,
    )
    from htc.evaluation.evaluate_superpixels import EvaluateSuperpixelImage
    from htc.evaluation.metrics.ECELoss import ECELoss
    from htc.evaluation.metrics.scores import normalize_grouped_cm
    from htc.fonts.set_font import set_font
    from htc.model_processing.ImageConsumer import ImageConsumer
    from htc.model_processing.Runner import Runner
    from htc.model_processing.TestLeaveOneOutPredictor import TestLeaveOneOutPredictor
    from htc.model_processing.TestPredictor import TestPredictor
    from htc.model_processing.ValidationPredictor import ValidationPredictor
    from htc.models.common.class_weights import calculate_class_weights
    from htc.models.common.distance_correlation import distance_correlation, distance_correlation_features
    from htc.models.common.ForwardHookPromise import ForwardHookPromise
    from htc.models.common.HierarchicalSampler import HierarchicalSampler
    from htc.models.common.HTCDataset import HTCDataset
    from htc.models.common.HTCDatasetStream import HTCDatasetStream
    from htc.models.common.HTCLightning import HTCLightning
    from htc.models.common.HTCModel import HTCModel
    from htc.models.common.MetricAggregation import MetricAggregation
    from htc.models.common.StreamDataLoader import StreamDataLoader
    from htc.models.common.torch_helpers import (
        FlexibleIdentity,
        copy_sample,
        cpu_only_tensor,
        move_batch_gpu,
        pad_tensors,
        smooth_one_hot,
        str_to_dtype,
    )
    from htc.models.common.utils import get_n_classes, infer_swa_lr, samples_equal
    from htc.models.data.DataSpecification import DataSpecification
    from htc.models.data.SpecsGeneration import SpecsGeneration
    from htc.models.image.DatasetImage import DatasetImage
    from htc.models.image.DatasetImageBatch import DatasetImageBatch
    from htc.models.image.DatasetImageStream import DatasetImageStream
    from htc.models.image.LightningImage import LightningImage
    from htc.models.image.ModelImage import ModelImage
    from htc.models.patch.DatasetPatchImage import DatasetPatchImage
    from htc.models.patch.DatasetPatchStream import DatasetPatchStream
    from htc.models.patch.LightningPatch import LightningPatch
    from htc.models.pixel.DatasetPixelStream import DatasetPixelStream
    from htc.models.pixel.LightningPixel import LightningPixel
    from htc.models.pixel.ModelPixel import ModelPixel
    from htc.models.pixel.ModelPixelRGB import ModelPixelRGB
    from htc.models.superpixel_classification.DatasetSuperpixelImage import DatasetSuperpixelImage
    from htc.models.superpixel_classification.DatasetSuperpixelStream import DatasetSuperpixelStream
    from htc.models.superpixel_classification.LightningSuperpixelClassification import LightningSuperpixelClassification
    from htc.models.superpixel_classification.ModelSuperpixelClassification import ModelSuperpixelClassification
    from htc.tissue_atlas.median_pixel.DatasetMedianPixel import DatasetMedianPixel
    from htc.tissue_atlas.median_pixel.LightningMedianPixel import LightningMedianPixel
    from htc.tissue_atlas.MetricAggregationClassification import MetricAggregationClassification
    from htc.tissue_atlas.settings_atlas import settings_atlas
    from htc.tivita.colorscale import tivita_colorscale
    from htc.tivita.DataPath import DataPath
    from htc.tivita.DatasetSettings import DatasetSettings
    from htc.tivita.hsi import read_tivita_dark, read_tivita_hsi, tivita_wavelengths
    from htc.tivita.metadata import generate_metadata_table, read_meta_file
    from htc.tivita.rgb import hsi_to_rgb, read_tivita_rgb
    from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
    from htc.utils.blosc_compression import compress_file, decompress_file
    from htc.utils.ColorcheckerReader import ColorcheckerReader
    from htc.utils.ColoredFileLog import ColoredFileLog
    from htc.utils.colors import generate_distinct_colors
    from htc.utils.Config import Config
    from htc.utils.DatasetDir import DatasetDir
    from htc.utils.DelayedFileHandler import DelayedFileHandler
    from htc.utils.DomainMapper import DomainMapper
    from htc.utils.DuplicateFilter import DuplicateFilter
    from htc.utils.general import (
        apply_recursive,
        clear_directory,
        merge_dicts_deep,
        safe_copy,
        sha256_file,
        subprocess_run,
    )
    from htc.utils.helper_functions import (
        basic_statistics,
        group_median_spectra,
        median_table,
        sort_labels,
        sort_labels_cm,
        utilization_table,
    )
    from htc.utils.LabelMapping import LabelMapping
    from htc.utils.LDA import LDA
    from htc.utils.MeasureTime import MeasureTime
    from htc.utils.MultiPath import MultiPath
    from htc.utils.parallel import p_imap, p_map
    from htc.utils.SpectrometerReader import SpectrometerReader
    from htc.utils.sqldf import sqldf
    from htc.utils.type_from_string import type_from_string
    from htc.utils.unify_path import unify_path
    from htc.utils.visualization import (
        add_std_fill,
        compress_html,
        create_class_scores_figure,
        create_confusion_figure,
        create_confusion_figure_comparison,
        create_ece_figure,
        create_overview_document,
        create_running_metric_plot,
        create_segmentation_overlay,
        create_surface_dice_plot,
        create_training_stats_figure,
        prediction_figure_html,
        show_loss_chart,
        visualize_dict,
    )
else:
    spec = importlib.util.find_spec("htc")
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects={"settings": settings, "settings_seg": settings_seg},
    )
    sys.modules[__name__].__spec__ = spec
