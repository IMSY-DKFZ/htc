# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pandas as pd
import torch
from rich.progress import track

from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.torch_helpers import move_batch_gpu
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import checkpoint_path


class InferenceTimeMeasurer:
    def __init__(self):
        path = DataPath.from_image_name("P043#2019_12_20_12_38_35")
        run_dir = (
            settings.training_dir
            / f"image/{settings_seg.model_comparison_timestamp}_generated_default_model_comparison"
        )
        config = Config(run_dir / "config.json")
        dataset = DatasetImage([path], train=False, config=config)

        # Load model for each fold (for ensembling)
        self.models = {}
        for fold_dir in sorted(run_dir.glob("fold*")):
            ckpt_file, _ = checkpoint_path(fold_dir)

            LightningClass = HTCLightning.class_from_config(config)
            model = LightningClass.load_from_checkpoint(
                ckpt_file, dataset_train=None, datasets_val=[dataset], config=config
            )
            model.eval()
            model.cuda()

            self.models[fold_dir] = model

        sample = dataset[0]
        self.batch = move_batch_gpu(sample)
        self.batch["features"] = self.batch["features"].unsqueeze(dim=0)

    @torch.autocast("cuda")
    @torch.no_grad()
    def inference_image(self) -> None:
        fold_predictions = []
        for model in self.models.values():
            fold_predictions.append(model.predict_step(self.batch)["class"].softmax(dim=1).float())

        torch.stack(fold_predictions).mean(dim=0)

    def warmup(self, n_steps: int = 10) -> None:
        for _ in range(n_steps):
            self.inference_image()

    def measure_time(self, n_steps: int = 100) -> list[float]:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        timings = []
        for _ in track(range(n_steps)):
            starter.record()
            self.inference_image()
            ender.record()
            torch.cuda.synchronize()

            timings.append(starter.elapsed_time(ender))

        return timings


if __name__ == "__main__":
    tm = InferenceTimeMeasurer()
    tm.warmup()
    timings = tm.measure_time()

    df = pd.DataFrame({"iteration": list(range(len(timings))), "time [ms]": timings})
    target_dir = settings.results_dir / "misc"
    target_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(target_dir / "inference_times.pkl.xz")
