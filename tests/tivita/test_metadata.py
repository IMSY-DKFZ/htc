# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.metadata import generate_metadata_table, read_meta_file, read_meta_patient


def test_metadata_table() -> None:
    paths = list(DataPath.iterate(settings.data_dirs.semantic))
    df = generate_metadata_table(paths)

    assert len(paths) == len(df)

    # Check some example types
    assert df["Fremdlichterkennung_Fremdlicht erkannt?"].dtype == bool
    assert df["Fremdlichterkennung_Intensity Grenzwert"].dtype == np.float64
    assert df["Camera_Exposure"].dtype == int

    assert "image_name" in df.columns
    assert "subject_name" in df.columns and "timestamp" in df.columns

    for p in paths:
        row = df.query(f'image_name == "{p.image_name()}"')
        assert len(row) == 1

        image_name_constructed = "#".join([row[name].item() for name in p.image_name_parts()])
        assert image_name_constructed == row["image_name"].item() and image_name_constructed == p.image_name()

    paths = [DataPath(settings.data_dirs.studies / "2022_09_29_Surgery2_baseline" / "2022_09_29_17_04_13")]
    df = generate_metadata_table(paths)
    assert len(df) == 1
    assert df["PatientID"].values == "calibration_white"


def test_read_meta_file() -> None:
    path = DataPath.from_image_name("P068#2020_07_20_18_17_26")
    meta = read_meta_file(path.camera_meta_path())

    assert meta["Camera_CamID"] == "0102-00085"
    assert meta == path.read_camera_meta()

    path_old = DataPath.from_image_name("P001#2018_07_26_12_13_48")
    assert path_old.read_camera_meta() is None


def test_read_meta_patient() -> None:
    path = settings.data_dirs.studies / "2022_09_29_Surgery2_baseline" / "2022_09_29_17_04_13" / "calibration_white.xml"
    meta = read_meta_patient(path)
    meta_gt = {
        "PatientID": "calibration_white",
        "PatientName": "^",
        "PatientBirthDate": "00000000",
        "PatientSex": "M",
        "AdmissionID": None,
        "AccessionNumber": None,
        "Requested Procedure Description": None,
        "Requested Procedure ID": None,
        "Patient Comments": None,
        "Performing Physician's Name": None,
        "Study ID": "2",
        "Study Date": "2022_09_",
        "Study Time": "17_04_13",
        "Series Number": "5",
        "Acquisition Date": "2022_09_",
        "Acquisition Time": "17_04_13",
        "Modality": "OT",
        "Manufacturer": "Diaspective-Vision",
        "DeviceSerialNumber": "0615-00023",
        "SoftwareVersions": "1.0.4",
        "SeriesDescription": "HSI-Measurement",
        "StudyDescription": "HSI-Measurement",
    }
    assert meta == meta_gt

    dpath = DataPath(settings.data_dirs.studies / "2022_09_29_Surgery2_baseline" / "2022_09_29_17_04_13")
    meta_path = dpath.patient_meta_path()
    assert meta_path == path

    assert dpath.meta("PatientID") == "calibration_white"

    meta2 = dpath.read_patient_meta()
    assert meta2 == meta_gt
