# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
from matplotlib.colors import to_rgb

from htc.cpp import nunique
from htc.utils.import_extra import requires_extra
from htc.utils.LabelMapping import LabelMapping

try:
    import nrrd

    _missing_library = ""
except ImportError:
    _missing_library = "nrrd"


@requires_extra(_missing_library)
def nrrd_mask(nrrd_file: Path) -> dict[str, Union[np.ndarray, LabelMapping]]:
    """
    Read an nrrd mask file from MITK. This file contains all the information from the annotation process.

    >>> from htc.tivita.DataPath import DataPath
    >>> path = DataPath.from_image_name("P065#2020_06_19_21_02_33")
    >>> mitk_data = nrrd_mask(path() / "annotations/2020_06_19_21_02_33#semantic#annotator5.nrrd")
    >>> np.unique(mitk_data["mask"])
    array([1, 2], dtype=uint8)
    >>> mitk_data["label_mapping"].index_to_name(1)
    'stomach'

    The "Exterior" in MITK always has the label index 0 and means that pixels are not labelled and are always considered invalid:
    >>> mitk_data["label_mapping"].name_to_index("unlabeled")
    0
    >>> mitk_data["label_mapping"].is_index_valid(0)
    False

    With the mitk_data you can easily map the segmentation to the desired target mapping:
    >>> from htc.settings_seg import settings_seg
    >>> mask = settings_seg.label_mapping.map_tensor(mitk_data["mask"], mitk_data["label_mapping"])
    >>> np.unique(mask)
    array([0, 6], dtype=uint8)
    >>> settings_seg.label_mapping.index_to_name(6)
    'stomach'

    In case of multi-layer NRRD files, an additional dimension is inserted at the front corresponding to the MITK layers:
    >>> path = DataPath.from_image_name("SPACE_000069#2020_11_05_11_43_51")
    >>> mitk_data = nrrd_mask(path() / "annotations/2020_11_05_11_43_51#semantic#primary.nrrd")
    >>> mitk_data["mask"].shape
    (4, 480, 640)

    Args:
        nrrd_file: Path to the nrrd file.

    Returns: Dictionary with the following content:
        - mask: Array with the raw label indices per pixel.
        - label_mapping: Label mapping to interpret the label indices.
    """
    data, header = nrrd.read(nrrd_file)

    total_n_labels = 0  # to be populated with number of labels across the layers

    mask = data.squeeze()
    mask = mask.T.astype(np.uint8)

    if mask.ndim == 3:
        mask = mask.transpose(2, 0, 1)

    mapping_nrrd = {}
    max_label_index = 0  # used to keep track of iterating label index in different layers. MITK assigns labels starting from 0 in each layer.

    # new MITK version NRRD files have to handled separately as they contain JSON meta data
    if "org.mitk.multilabel.segmentation.version" in header:
        label_groups = json.loads(header["org.mitk.multilabel.segmentation.labelgroups"])
        n_layers = len(label_groups)

        # in the new format there is no exterior label, so the total n labels are incremented here
        total_n_labels += 1
        mapping_nrrd["unlabeled"] = 0

        for layer in range(n_layers):

            if label_groups[layer]["labels"] is not None:
                total_n_labels += len(label_groups[layer]["labels"])
            else:
                label_groups[layer]["labels"] = []

            if mask.ndim == 3:
                # MITK assigns the label index for each layer individually according to the order in which the annotation was performed. This leads to different label indices for the same class in different layers. Therefore, a remapping is performed using the label_index of the previous layer(s).
                layer_mask = deepcopy(mask[layer, :, :])  # needed for remapping
            else:
                layer_mask = mask

            for label in label_groups[layer]["labels"]:
                label_name = label["name"]
                label_index = label["value"]

                # in case the label name has the label order number as a prefix e.g. 12_kidney, then extract the label name
                match = re.search(r"^\d+_", label_name)
                if match is not None:
                    label_name = label_name.removeprefix(match.group(0))

                if label_name not in mapping_nrrd:
                    mapping_nrrd[label_name] = (
                        max(mapping_nrrd.values()) + 1
                    )  # remapping to the smallest unassigned label_index

                if mask.ndim == 3:
                    mask[layer, layer_mask == label_index] = mapping_nrrd[label_name]
                else:
                    mask[layer_mask == label_index] = mapping_nrrd[label_name]

            max_label_index = max(mapping_nrrd.values())
    else:
        n_layers = int(header["layers"])

        for layer in range(n_layers):
            n_labels = int(header[f"layer_00{layer}"])
            total_n_labels += n_labels

            if mask.ndim == 3:
                layer_mask = deepcopy(mask[layer, :, :])
            else:
                layer_mask = mask

            for i in range(n_labels):
                root = ET.fromstring(header[f"org.mitk.label_00{layer}_{i:05d}"].replace("\\n", "\n"))
                label_index = int(root.find("property[@key='value']/unsigned").attrib["value"])

                label_name = root.find("property[@key='name']/string").attrib["value"]
                match = re.search(r"^\d+_", label_name)
                if match is not None:
                    label_name = label_name.removeprefix(match.group(0))

                if i == 0:
                    mapping_nrrd["unlabeled"] = label_index
                else:
                    if label_name not in mapping_nrrd:
                        mapping_nrrd[label_name] = max(mapping_nrrd.values()) + 1

                    if mask.ndim == 3:
                        mask[layer, layer_mask == label_index] = mapping_nrrd[label_name]
                    else:
                        mask[layer_mask == label_index] = mapping_nrrd[label_name]

            max_label_index = max(mapping_nrrd.values())

    mappings_nrrd = LabelMapping(mapping_nrrd, last_valid_label_index=max_label_index, zero_is_invalid=True)

    assert nunique(mask) <= total_n_labels

    return {"mask": mask, "label_mapping": mappings_nrrd}


@requires_extra(_missing_library)
def segmentation_to_nrrd(
    nrrd_file: Path,
    mask: np.ndarray,
    mapping_mask: LabelMapping,
) -> None:
    """
    Converts an existing segmentation mask to an nrrd file which can be read by MITK. This is useful if existing masks should be loaded into MITK for visualization or adaptations.

    >>> from htc.tivita.DataPath import DataPath
    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile() as tmpfile:
    ...    tmpfile = Path(tmpfile.name)
    ...    path = DataPath.from_image_name("SPACE_000001#2020_08_14_11_11_22")
    ...    segmentation_dict = path.read_segmentation(annotation_name="all")
    ...    mask = np.stack(list(segmentation_dict.values()))
    ...    segmentation_to_nrrd(nrrd_file=tmpfile, mask=mask, mapping_mask=LabelMapping.from_path(path))
    ...    labels = nrrd_mask(nrrd_file=tmpfile)['label_mapping'].label_names()
    >>> labels
    ['colon', 'omentum', 'small_bowel', 'fat', 'instrument', 'background', 'blue_cloth', 'unclear_organic', 'tag_blood']

    Args:
        nrrd_file: Path where the nrrd file should be stored.
        mask: a dict of masks, each key representing an annotation name e.g. {{annotation_name1: mask, annotation_name2: mask...}}. If None, path must be given.
        mapping_mask: Label mapping for the segmentation mask which gives every label index in the segmentation mask a name. If None, path must be given.
    """

    # create a copy of mask
    mask = deepcopy(mask)

    invalid_pixels = ~mapping_mask.is_index_valid(mask)

    # We need to remap the labels to consecutive values starting from 1 because 0 will be the unlabeled pixels in MITK
    mapping_mitk = {"Exterior": 0}
    i = 1
    for label_index in np.unique(mask):
        if mapping_mask.is_index_valid(label_index):
            mapping_mitk[mapping_mask.index_to_name(label_index)] = i
            i += 1
        else:
            mapping_mitk[mapping_mask.index_to_name(label_index)] = 0

    mapping_mitk = LabelMapping(mapping_mitk, zero_is_invalid=True)
    assert mapping_mitk.last_valid_label_index == i - 1
    assert len(mapping_mitk) <= len(mapping_mask)

    # Remap segmentation to a valid MITK mask
    mapping_mitk.map_tensor(mask, mapping_mask)
    n_labels = len(mapping_mitk.label_names(include_invalid=True))
    assert nunique(mask) <= n_labels
    assert np.all(mask[invalid_pixels] == 0), "All invalid pixels should have been mapped to 0"

    # MITK/nrrd loads the image as (width, height)

    if mask.ndim == 3:
        n_layers = mask.shape[0]
        mask = np.expand_dims(np.transpose(mask, axes=(0, 2, 1)), -1)
    else:
        n_layers = 1
        mask = np.expand_dims(mask.T, -1)

    def mitk_label_header(label_index: int, label_name: str, label_color: str) -> dict:
        # 0 = background/invalid in MITK
        opacity = 0.600000024 if label_index != 0 else 0
        locked = True if label_index != 0 else False
        r, g, b = to_rgb(label_color)

        meta = {
            "color": {"type": "ColorProperty", "value": [float(r), float(g), float(b)]},
            "locked": locked,
            "name": label_name,
            "opacity": opacity,
            "value": int(label_index),
            "visible": True,
        }

        return meta

    header = {
        "modality": "org.mitk.multilabel.segmentation",
        "DICOM_0008_0060": '{"values":[{"z":0, "t":0, "value":"SEG"}]}',
        "DICOM_0008_103E": '{"values":[{"z":0, "t":0, "value":"MITK Segmentation"}]}',
        "org.mitk.multilabel.segmentation.labelgroups": [],
        "org.mitk.multilabel.segmentation.unlabeledlabellock": "0",
        "org.mitk.multilabel.segmentation.version": "1",
        "type": "unsigned short",
        "space": "left-posterior-superior",
        "space origin": [0, 0, 0],
    }

    if n_layers == 1:
        header["space directions"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        header["kinds"] = ["domain", "domain", "domain"]
    else:
        header["space directions"] = [[np.nan, np.nan, np.nan], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        header["kinds"] = ["vector", "domain", "domain", "domain"]

    # switching back to MITK increasing order of label index
    curr_label_index = 1

    mask_copy = deepcopy(mask)

    for layer_index in range(n_layers):
        labelgroup = {"labels": []}

        if mask.ndim == 4:
            label_indices = np.unique(mask[layer_index, ...])
        else:
            label_indices = mapping_mitk.label_indices(include_invalid=True)

        for label_index in label_indices:
            if label_index == 0:
                continue

            mask_copy[layer_index, ...][mask[layer_index, ...] == label_index] = curr_label_index

            labelgroup["labels"].append(
                mitk_label_header(
                    curr_label_index, mapping_mitk.index_to_name(label_index), mapping_mitk.index_to_color(label_index)
                )
            )

            curr_label_index += 1
        header["org.mitk.multilabel.segmentation.labelgroups"].append(labelgroup)

    header["org.mitk.multilabel.segmentation.labelgroups"] = json.dumps(
        header["org.mitk.multilabel.segmentation.labelgroups"]
    )

    nrrd.write(str(nrrd_file), data=mask_copy.astype(np.ushort), header=header)


def segmentation_to_nrrd_annotation_name(
    nrrd_file: Path,
    mask: dict[str, np.ndarray],
    mapping_mask: LabelMapping,
    annotation_name_to_layer: dict[str, int] = None,
) -> None:
    """
    Converts an existing segmentation mask to an nrrd file which can be read by MITK. This is useful if existing masks should be loaded into MITK for visualization or adaptations.
    This function can be used to directly convert a dictionary of masks read from the path.

    >>> from htc.tivita.DataPath import DataPath
    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile() as tmpfile:
    ...    tmpfile = Path(tmpfile.name)
    ...    path = DataPath.from_image_name("SPACE_000001#2020_08_14_11_11_22")
    ...    mask = path.read_segmentation(annotation_name="all")
    ...    segmentation_to_nrrd_annotation_name(nrrd_file=tmpfile, mask=mask, mapping_mask=LabelMapping.from_path(path), annotation_name_to_layer={"semantic#primary": 0, "polygon#annotator1": 1})
    ...    labels = nrrd_mask(nrrd_file=tmpfile)['label_mapping'].label_names()
    >>> labels
    ['background', 'blue_cloth', 'colon', 'omentum', 'small_bowel', 'unclear_organic']

    Args:
        nrrd_file: Path where the nrrd file should be stored.
        mask: a dict of masks, each key representing an annotation name e.g. {{annotation_name1: mask, annotation_name2: mask...}}. If None, path must be given.
        mapping_mask: Label mapping for the segmentation mask which gives every label index in the segmentation mask a name. If None, path must be given.
        annotation_name_to_layer: Maps annotation names to layers in MITK. Layers must be integers and define the order of the segmentation masks in MITK. The dictionary has the form: `{annotation_name: layer_index}`
    """
    mask = deepcopy(mask)

    # take annotation names from annotation_name_to_layer attribute
    # if the annotation_name_to_layer is None, then the default annotation names are used
    # use all of the annotation names from the mask if annotation_name_to_layer is not set
    if annotation_name_to_layer is not None:
        annotation_names = list(annotation_name_to_layer.keys())
    else:
        annotation_names = mask.keys()

    assert (
        type(mask) == dict
    ), "The mask has to be dict containing all annotations, of the form: `{annotation_name: layer_index}`"

    stacked_masks = []

    for annotation_name in annotation_names:
        assert annotation_name in mask, f"Request annotation name {annotation_name} not present in mask"
        stacked_masks.append(mask[annotation_name])

    mask = np.stack(stacked_masks)

    segmentation_to_nrrd(nrrd_file=nrrd_file, mask=mask, mapping_mask=mapping_mask)
