# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc_projects.camera.settings_camera import settings_camera


class TestLabelMapping:
    def test_basics(self) -> None:
        m = LabelMapping({"a": 0, "b": 1, "a_second_name": 0, "ignored": 2}, last_valid_label_index=1)
        assert m.name_to_index("a") == 0
        assert m.index_to_name(0) == "a"
        assert m.index_to_name(0, all_names=True) == ["a", "a_second_name"]
        assert len(m) == 2
        assert m.label_names() == ["a", "b"] and m.label_indices() == [0, 1]
        assert m.label_names(all_names=True) == ["a", "a_second_name", "b"]
        assert m.label_names(include_invalid=True) == ["a", "b", "ignored"] and m.label_indices(
            include_invalid=True
        ) == [0, 1, 2]
        assert list(m.mapping_name_index.keys()) == ["a", "a_second_name", "b", "ignored"]
        assert "a" in m and 1 in m
        assert "aa" not in m

        assert m.is_index_valid(0) and m.is_index_valid(1)
        assert not m.is_index_valid(2)
        assert torch.all(
            m.is_index_valid(torch.tensor([0, 1, 1, 2, 2]))
            == torch.tensor([True, True, True, False, False], dtype=torch.bool)
        )
        assert np.all(m.is_index_valid(np.array([0, 2]) == np.array([True, False], dtype=bool)))

        with pytest.raises(ValueError):
            1.1 in m  # noqa: B015
            {"x": 1} in m  # noqa: B015

        m2 = LabelMapping({"a": 10, "b": 20}, last_valid_label_index=20)
        tensor = torch.stack([
            torch.ones(100, 100, dtype=torch.int64) * 10,
            torch.ones(100, 100, dtype=torch.int64) * 20,
        ])
        m.map_tensor(tensor, m2)
        assert len(m2) == 2
        assert torch.all(tensor[0, ...] == 0)
        assert torch.all(tensor[1, ...] == 1)
        assert m2.label_names() == ["a", "b"] and m2.label_indices() == [10, 20]

        m3 = LabelMapping({"a": 10, "b": 20})
        assert len(m3) == 2
        assert m3.label_names() == ["a", "b"]
        assert m3.last_valid_label_index == 20

        m4 = LabelMapping({"a": 10, "b": 20, "c": settings.label_index_thresh})
        assert len(m4) == 2
        assert m4.label_names() == ["a", "b"]
        assert m4.last_valid_label_index == 20

    def test_colors(self) -> None:
        m = LabelMapping(
            {"a": 0, "b": 1, "a_second_name": 0, "ignored": 2},
            last_valid_label_index=1,
            label_colors={"a": "#FFFFFF", "invalid": "#AAAAAA"},
        )

        assert m.name_to_color("a") == "#FFFFFF" and m.name_to_color("a_second_name") == "#FFFFFF"
        assert m.name_to_color("ignored") == "#AAAAAA"
        assert m.index_to_color(0) == "#FFFFFF"
        assert m.index_to_color(2) == "#AAAAAA"
        with pytest.raises(ValueError):
            m.name_to_color("b")

        m2 = LabelMapping(
            {"a": 0, "b": 1, "a_second_name": 0, "ignored": 2},
            last_valid_label_index=1,
            label_colors={"a": "#FFFFFF", "a_second_name": "#EEEEEE", "invalid": "#AAAAAA"},
        )

        assert m2.name_to_color("a") == "#FFFFFF"
        assert m2.name_to_color("a_second_name") == "#EEEEEE"

        rename_dict = {"a": "new"}
        m.rename(rename_dict)
        assert "a" not in m.mapping_name_index.keys()
        assert "new" in m.mapping_name_index.keys()
        assert "a" not in m.label_colors.keys()
        assert "new" in m.label_colors.keys()
        assert "a" not in m.mapping_index_name.values()
        assert "new" in m.mapping_index_name.values()

    def test_sorted(self) -> None:
        m = LabelMapping({"a": 2, "b": 1, "x": 0, "a_second_name": 0}, last_valid_label_index=2)
        assert len(m) == 3
        assert m.label_names() == ["x", "b", "a"]
        assert list(m.mapping_name_index.keys()) == ["x", "a_second_name", "b", "a"]
        assert m.label_indices() == [0, 1, 2]

    def test_from_settings_seg(self) -> None:
        m = settings_seg.label_mapping
        assert m.name_to_index("background") == 0
        assert m.name_to_color("stomach") == settings.label_colors["stomach"]

    def test_from_path_from_config(self, tmp_path: Path) -> None:
        path = DataPath.from_image_name("P043#2019_12_20_12_38_35")
        mapping = LabelMapping.from_path(path)
        for label_name, label_index in path.dataset_settings["label_mapping"].items():
            assert mapping.name_to_index(label_name) == label_index

        sample_none = DatasetImage([path], train=False, config=Config({"label_mapping": None}))[0]
        config_mapping = Config({"label_mapping": mapping})
        config_mapping.save_config(tmp_path / "config.json")
        sample_mapping = DatasetImage([path], train=False, config=config_mapping)[0]

        assert torch.all(sample_none["labels"] == sample_mapping["labels"])
        assert torch.all(sample_none["valid_pixels"] == sample_mapping["valid_pixels"])

        config_loaded = Config(tmp_path / "config.json")
        mapping_loaded = LabelMapping.from_config(config_loaded)
        assert mapping_loaded == mapping

        sample_loaded = DatasetImage([path], train=False, config=config_loaded)[0]
        assert torch.all(sample_none["labels"] == sample_loaded["labels"])
        assert torch.all(sample_none["valid_pixels"] == sample_loaded["valid_pixels"])

    def test_from_config_to_json(self) -> None:
        config = Config({"label_mapping": {"a": 0, "b": 1, "c": settings.label_index_thresh}})
        mapping = LabelMapping.from_config(config)
        assert mapping.last_valid_label_index == 1
        assert mapping.label_indices() == [0, 1]
        assert mapping.label_names() == ["a", "b"]
        assert len(mapping) == 2

        json_dict = json.loads(json.dumps(config.data, cls=AdvancedJSONEncoder))
        assert json_dict == {
            "label_mapping": {
                "mapping_name_index": {"a": 0, "b": 1, "c": 100},
                "last_valid_label_index": 1,
                "zero_is_invalid": False,
                "unknown_invalid": True,
                "mapping_index_name": {"0": "a", "1": "b", "100": "c"},
            }
        }
        assert mapping == LabelMapping.from_config(json_dict)

        config = Config({"label_mapping": LabelMapping({"a": 0, "b": 1, "c": 2}, last_valid_label_index=1)})
        mapping = LabelMapping.from_config(config)
        assert mapping.last_valid_label_index == 1
        assert mapping.label_indices() == [0, 1]
        assert mapping.label_names() == ["a", "b"]
        assert len(mapping) == 2

        config = Config({"label_mapping": "htc.settings_seg>label_mapping"})
        mapping_config = LabelMapping.from_config(config)
        mapping_settings = settings_seg.label_mapping
        assert mapping_config == mapping_settings

        config = Config({"label_mapping": "htc_projects.camera.settings_camera>label_mapping"})
        mapping_config = LabelMapping.from_config(config)
        assert settings_camera.label_mapping == mapping_config
        assert isinstance(config["label_mapping"], LabelMapping)

    def test_from_data_dir(self) -> None:
        path = DataPath.from_image_name("P043#2019_12_20_12_38_35")
        mapping_path = LabelMapping.from_path(path)
        mapping_data_dir = LabelMapping.from_data_dir(settings.data_dirs.semantic)
        assert mapping_path == mapping_data_dir

    def test_image_mapping(self) -> None:
        dataset = DatasetImage.example_dataset()  # Uses the default label mapping from the settings
        labels = dataset[0]["labels"]

        assert all(
            v in settings_seg.label_mapping for v in labels.unique() if settings_seg.label_mapping.is_index_valid(v)
        )

    def test_zero_invalid(self) -> None:
        m = LabelMapping(
            {"background": 0, "b": 1, "b_second_name": 1, "ignored": 2}, last_valid_label_index=1, zero_is_invalid=True
        )
        assert m.label_names() == ["b"]
        assert m.label_names(all_names=True) == ["b", "b_second_name"]
        assert torch.all(m.is_index_valid(torch.tensor([0, 1, 2])) == torch.tensor([False, True, False]))

    def test_unknown_invalid(self) -> None:
        m = LabelMapping(
            {"background": 0, "b": 1, "b_second_name": 1, "ignored": 2},
            last_valid_label_index=1,
        )
        assert m.label_names() == ["background", "b"]
        assert m.label_names(all_names=True) == ["background", "b", "b_second_name"]
        assert torch.all(m.is_index_valid(torch.tensor([0, 1, 2])) == torch.tensor([True, True, False]))
        assert m.name_to_index("some_unknown_name") == 3
        assert m.name_to_index("another_unknown_name") == 3
        assert m.index_to_name(2) == "ignored"
        assert m.index_to_name(3) == "unknown"

    def test_append(self) -> None:
        m = LabelMapping({"x": 0, "b": 1, "a_second_name": 0}, last_valid_label_index=1)
        assert m.label_names() == ["x", "b"]

        m.append("c")
        assert m.label_names() == ["x", "b", "c"]
        assert m.last_valid_label_index == 2

        m = LabelMapping({"x": 0, "b": 1, "a_second_name": 0, "invalid": 2}, last_valid_label_index=1)
        m.append("c")
        assert m.label_names() == ["x", "b", "c"]
        assert m.last_valid_label_index == 2
        assert m.mapping_index_name == {0: "x", 1: "b", 2: "c", 3: "invalid"}
        assert m.mapping_name_index == {"x": 0, "a_second_name": 0, "b": 1, "c": 2, "invalid": 3}

        m = LabelMapping({"x": 0, "b": 1, "a_second_name": 0, "invalid": 2}, last_valid_label_index=1)
        m.append("invalid2", invalid=True)
        assert m.label_names() == ["x", "b"]
        assert m.last_valid_label_index == 1
        assert m.mapping_index_name == {0: "x", 1: "b", 2: "invalid", 3: "invalid2"}
        assert m.mapping_name_index == {"x": 0, "a_second_name": 0, "b": 1, "invalid": 2, "invalid2": 3}

        m = LabelMapping({"x": 0, "b": 1, "a_second_name": 0, "invalid": 10}, last_valid_label_index=1)
        m.append("invalid2", invalid=True)
        assert m.label_names() == ["x", "b"]
        assert m.last_valid_label_index == 1
        assert m.mapping_index_name == {0: "x", 1: "b", 10: "invalid", 2: "invalid2"}
        assert m.mapping_name_index == {"x": 0, "a_second_name": 0, "b": 1, "invalid": 10, "invalid2": 2}
