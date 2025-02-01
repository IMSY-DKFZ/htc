# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import io
import json
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib.colors import to_hex, to_rgb
from typing_extensions import Buffer

from htc.cpp import nunique
from htc.utils.import_extra import requires_extra
from htc.utils.LabelMapping import LabelMapping

try:
    import nrrd

    _missing_library = ""
except ImportError:
    _missing_library = "nrrd"


# keeping track of relevant header attribute names, so that these variables can be used in other scripts
# related to reading and analyzing of nrrd files
header_attribute_version = "org.mitk.multilabel.segmentation.version"
header_attribute_labelgroups = "org.mitk.multilabel.segmentation.labelgroups"


@requires_extra(_missing_library)
def nrrd_mask(nrrd_file: Path) -> dict[str, np.ndarray | LabelMapping]:
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
        - labels_per_layer: A list containing label names present in each layer of the nrrd file
        - conflicting_label_found: A boolean which indicates if a label value conflict i.e. same label value specified more than once in the nrrd file metadata, is found
        - layer_names: Optional list of names for the layers. This key is only present if the information is contained in the NRRD file.
    """
    data, header = nrrd.read(nrrd_file)

    total_n_labels = 0  # to be populated with number of labels across the layers

    mask = data.squeeze()
    mask = mask.T.astype(np.uint8)

    if mask.ndim == 3:
        mask = mask.transpose(2, 0, 1)

    mapping_nrrd = {}
    mapping_nrrd_colors = {}
    max_label_index = 0  # used to keep track of iterating label index in different layers. MITK assigns labels starting from 0 in each layer.

    # keep track of labels within each layer in the nrrd metadata. This is useful for validating nrrd files.
    labels_per_layer = []
    layer_names = []

    # keep track of all indexes and the corresponding labels in the nrrd file. This is useful for validating nrrd files
    index_to_label_layers = {}
    conflicting_label_found = False

    # new MITK version NRRD files have to handled separately as they contain JSON meta data
    if header_attribute_version in header:
        label_groups = json.loads(header[header_attribute_labelgroups])
        n_layers = len(label_groups)

        # in the new format there is no exterior label, so the total n labels are incremented here
        total_n_labels += 1
        mapping_nrrd["unlabeled"] = 0

        for layer in range(n_layers):
            labels_per_layer.append([])
            if label_groups[layer]["labels"] is not None:
                total_n_labels += len(label_groups[layer]["labels"])
            else:
                label_groups[layer]["labels"] = []

            if "name" in label_groups[layer]:
                layer_names.append(label_groups[layer]["name"])

            if mask.ndim == 3:
                # MITK assigns the label index for each layer individually according to the order in which the annotation was performed. This leads to different label indices for the same class in different layers. Therefore, a remapping is performed using the label_index of the previous layer(s).
                layer_mask = deepcopy(mask[layer, :, :])  # needed for remapping
            else:
                layer_mask = mask

            for label in label_groups[layer]["labels"]:
                label_name = label["name"]
                label_index = label["value"]
                label_color = to_hex(label["color"]["value"])

                # in case the label name has the label order number as a prefix e.g. 12_kidney, then extract the label name
                match = re.search(r"^\d+_", label_name)
                if match is not None:
                    label_name = label_name.removeprefix(match.group(0))

                labels_per_layer[layer].append(label_name)

                if label_index in index_to_label_layers:
                    conflicting_label_found = True

                index_to_label_layers.update({label_index: label_name})
                if label_name not in mapping_nrrd:
                    # remapping to the smallest unassigned label_index
                    mapping_nrrd[label_name] = max(mapping_nrrd.values()) + 1
                    mapping_nrrd_colors[label_name] = label_color

                if mask.ndim == 3:
                    mask[layer, layer_mask == label_index] = mapping_nrrd[label_name]
                else:
                    mask[layer_mask == label_index] = mapping_nrrd[label_name]

            max_label_index = max(mapping_nrrd.values())
    else:
        n_layers = int(header["layers"])

        for layer in range(n_layers):
            labels_per_layer.append([])
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

                labels_per_layer[layer].append(label_name)

                if label_index in index_to_label_layers:
                    conflicting_label_found = True

                index_to_label_layers.update({label_index: label_name})
                colors = root.find("property[@key='color']/color").attrib
                label_color = to_hex((float(colors["r"]), float(colors["g"]), float(colors["b"])))

                if i == 0:
                    mapping_nrrd["unlabeled"] = label_index
                    mapping_nrrd_colors["unlabeled"] = label_color
                else:
                    if label_name not in mapping_nrrd:
                        mapping_nrrd[label_name] = max(mapping_nrrd.values()) + 1
                        mapping_nrrd_colors[label_name] = label_color

                    if mask.ndim == 3:
                        mask[layer, layer_mask == label_index] = mapping_nrrd[label_name]
                    else:
                        mask[layer_mask == label_index] = mapping_nrrd[label_name]

            max_label_index = max(mapping_nrrd.values())

    mappings_nrrd = LabelMapping(
        mapping_nrrd, last_valid_label_index=max_label_index, zero_is_invalid=True, label_colors=mapping_nrrd_colors
    )

    assert nunique(mask) <= total_n_labels

    res = {
        "mask": mask,
        "label_mapping": mappings_nrrd,
        "labels_per_layer": labels_per_layer,
        "conflicting_label_found": conflicting_label_found,
    }
    if len(layer_names) > 0:
        res["layer_names"] = layer_names

    return res


@requires_extra(_missing_library)
def segmentation_to_nrrd(
    nrrd_file: Path,
    mask: np.ndarray,
    mapping_mask: LabelMapping,
    mask_labels_only: bool = True,
    layer_names: list[str] = None,
) -> None:
    """
    Converts an existing segmentation mask to an nrrd file which can be read by MITK. This is useful if existing masks should be loaded into MITK for visualization or adaptations.

    >>> from htc.tivita.DataPath import DataPath
    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile() as tmpfile:
    ...     tmpfile = Path(tmpfile.name)
    ...     path = DataPath.from_image_name("SPACE_000001#2020_08_14_11_11_22")
    ...     segmentation_dict = path.read_segmentation(annotation_name="all")
    ...     mask = np.stack(list(segmentation_dict.values()))
    ...     segmentation_to_nrrd(nrrd_file=tmpfile, mask=mask, mapping_mask=LabelMapping.from_path(path))
    ...     labels = nrrd_mask(nrrd_file=tmpfile)["label_mapping"].label_names()
    >>> labels
    ['colon', 'omentum', 'small_bowel', 'fat', 'instrument', 'background', 'blue_cloth', 'unclear_organic', 'tag_blood']

    Args:
        nrrd_file: Path where the nrrd file should be stored.
        mask: Mask data containing the annotations. If 2D, a single MITK layer will be created. If 3D, an MITK layer will be created per layer.
        mapping_mask: Label mapping for the segmentation mask which gives every label index in the segmentation mask a name. If None, path must be given.
        mask_labels_only: Ensures that the nrrd file only contains labels in the metadata that are found in the mask, as opposed to adding all labels in the mapping to the metadata.
        layer_names: Optional list with names for the layers in the mask. Those names will show up in MITK and usually correspond to the annotation names of the layers.
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

    # if not only the labels available in masks, but all labels should be added to the nrrd files' metadata
    if not mask_labels_only:
        missing_labels = sorted(set(mapping_mask.label_names()) - set(mapping_mitk.keys()))
        for label in missing_labels:
            mapping_mitk[label] = i
            i += 1

    mapping_mitk = LabelMapping(mapping_mitk, zero_is_invalid=True, label_colors=mapping_mask.label_colors)
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

    if layer_names is not None:
        assert len(layer_names) == n_layers, "The number of layer names must match the number of layers in the mask"

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
        header_attribute_labelgroups: [],
        "org.mitk.multilabel.segmentation.unlabeledlabellock": "0",
        header_attribute_version: "1",
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
        if layer_names is not None:
            labelgroup["name"] = layer_names[layer_index]

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
        header[header_attribute_labelgroups].append(labelgroup)

    header[header_attribute_labelgroups] = json.dumps(header[header_attribute_labelgroups])

    # Unfortunately, the pynrrd library writes the current timestamp into the file as comment which (for example) breaks git diffs
    # The solution here is to remove those comments manually from the NRRD file
    class NRRDBytesIO(io.BytesIO):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.write_call_counts = 0

        def write(self, buffer: Buffer) -> int:
            self.write_call_counts += 1
            if 2 <= self.write_call_counts <= 3:
                assert buffer.decode("ascii").startswith("#")
                return 0
            else:
                return super().write(buffer)

    io_stream = NRRDBytesIO()
    nrrd.write(io_stream, data=mask_copy.astype(np.ushort), header=header)

    with nrrd_file.open("wb") as f:
        f.write(io_stream.getbuffer())
