# Custom `DataPath` class

A fundamental concept of the htc framework is the `DataPath` class which is the main interaction point with all image-based actions like reading data cubes, accessing image properties or loading annotations. This tutorial gives hinters if the default classes (like [`DataPathTivita`](../htc/tivita/DataPathTivita.py)) are not enough anymore and you want to write your own data path class. As a general hint, you can explore the existing classes in `../htc/tivita` for examples. In general, datasets are expected to be structured like

```text
YOUR_DATASET
├── data  # Your raw data and annotations
│   └── dataset_settings.json  # General information about your dataset with easy access in the framework (see below)
└── intermediates  # Generated files
```

It is not necessary to have this structure but maybe makes your life easier ;-)

In the following, some general important concepts are explained but feel free to overwrite any method in the `DataPath` class as you wish.

## Iterating

To adapt your data path class to your dataset structure, you can overwrite the default iterate method. Here, you can define the logic to find all images in your dataset and save custom attributes (for example, subject identifiers). Here you can also load your custom dataset settings (see below) and assign it to your data paths. A stub for the `iterate` looks as follows:

```python
class DataPathCustom(DataPath):
    @staticmethod
    def iterate(
        data_dir: Path,
        filters: list[Callable[["DataPath"], bool]],
        annotation_name: Union[str, list[str]],
    ) -> Iterator["DataPathCustom"]:
        if data_dir.name == "data":
            # Optional but recommended (see below)
            dataset_settings = DatasetSettings(data_dir / "dataset_settings.json")

            # Optional, only if needed/available
            intermediates_dir = settings.datasets.find_intermediates_dir(data_dir)

            # Adjust looping according to your dataset structure
            for image_dir in sorted(data_dir.iterdir()):
                # Add custom attributes as needed
                path = DataPathCustom(image_dir, data_dir, intermediates_dir=intermediates_dir, dataset_settings=dataset_settings, annotation_name_default=annotation_name)
                if all([f(path) for f in filters]):
                    yield path
        else:
            # Fallback to the default DataPathTivita class if the given data_dir is unknown (e.g. because the user requested a subdirectory and your class cannot handle iteration over subdirectories)
            yield from DataPathTivita.iterate(data_dir, filters, annotation_name)
```

## Annotations

It may be useful to make your annotations for an image available. In the case of segmentations (same spatial shape as the input image), you can overload the `read_segmentation()` method. It has no required arguments and expects to return a Numpy array containing the label indices for each pixel:

```python
def read_segmentation(self) -> np.ndarray:
    # Or relative to self.intermediates_dir if available
    return np.load(self.data_dir / "annotations.npz")  # e.g. np.uint8 with shape [480, 640]
```

## Dataset settings

It can be very useful to store information which apply to all your images at one place. For this, you can use the [`DatasetSettings`](../htc/tivita/DatasetSettings.py) class which basically loads a global JSON file with attributes for all your images. This object is created once and a reference is stored for each data path `path.dataset_settings` so that you can easily access the information. A typical dataset settings may look like

```json
{
    "dataset_name": "NAME_OF_YOUR_DATASET",
    "data_path_class": "htc.tivita.DataPathMultiorgan>DataPathMultiorgan",
    "shape": [480, 640, 100],
    "shape_names": ["height", "width", "channels"],
    "label_mapping": {
        "class_1": 0,
        "class_2": 1,
        "unlabeled": 255
    },
    "last_valid_label_index": 1
}
```

Of course, you can add custom information as you wish.

-   `label_mapping` is especially interesting if you also overload the `read_segmentation()` method. This basically gives a meaning to the labels in your segmentation mask (which can again be used for remapping tasks for training).
-   `data_path_class` is a very special key. Here you can specify a Python import (`module>class`). If your dataset settings is stored in `data_dir / "dataset_settings.json"` (i.e. in the top of your `data` directory), then you can just write
    ```python
    paths = list(DataPath.iterate(Path("your path")))
    ```
    and it will automatically pick up your custom class. This way, you can always use the `DataPath` class as general entry point.
