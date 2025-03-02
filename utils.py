import os
import random
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import yaml
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import (
    DKittyMorphologyDataset,
)
from design_bench.datasets.continuous.superconductor_dataset import (
    SuperconductorDataset,
)
from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
from design_bench.task import Task
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

_TKWARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}
_BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True
    tf.random.set_seed(seed)


def create_special_dataset_fast_unique(
    x: np.ndarray, y: np.ndarray, m: int = 1000, num_samples=10000
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) == 2, "X should be of shape (#data, input_size)"

    x_tensor = torch.from_numpy(x).to(**_TKWARGS)
    y_tensor = torch.from_numpy(y).to(**_TKWARGS)

    n, d = x_tensor.shape
    indices = torch.stack(
        [torch.randperm(n)[:m].cuda() for _ in tqdm(range(num_samples))]
    )

    train_X = x_tensor[indices]
    train_Y = y_tensor[indices]

    return train_X.detach().cpu(), train_Y.detach().cpu()


def build_data_loader(
    x: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    batch_size: int = 128,
    require_valid: bool = True,
    valid_ratio_if_valid: float = 0.2,
    drop_last: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:

    # assert len(x.shape) == 2, "X should be of shape (#data, input_size)"
    assert (
        not require_valid
    ) or 0 <= valid_ratio_if_valid <= 1, "valid ratio should be within [0, 1]"
    assert batch_size >= 1

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(**_TKWARGS)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).to(**_TKWARGS)

    train_dataset = TensorDataset(x, y)
    data_size = len(train_dataset)

    if require_valid:
        lengths = [
            data_size - int(valid_ratio_if_valid * data_size),
            int(valid_ratio_if_valid * data_size),
        ]
        train_dataset, validate_dataset = random_split(train_dataset, lengths)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )

    valid_loader = (
        DataLoader(
            dataset=validate_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
        )
        if require_valid
        else None
    )

    return (train_loader, valid_loader if require_valid else None)


def load_default_config(loss_type: Optional[str] = None) -> dict:
    config_path = os.path.join(_BASE_PATH, "config", "default.yaml")
    assert os.path.exists(config_path), f"config {config_path} not found"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if loss_type is None:
        loss_type = config["loss"]

    loss_config_path = os.path.join(_BASE_PATH, "config", "losses", f"{loss_type}.yaml")
    if not os.path.exists(loss_config_path):
        loss_config = None
    else:
        with open(loss_config_path, "r") as f:
            loss_config = yaml.load(f, Loader=yaml.FullLoader)

    if loss_config is None:
        loss_config = {}

    if "num_classes" in loss_config.keys():
        config["output_dim"] = loss_config["num_classes"]
    config["loss_config"] = loss_config

    return config


DATASETS = {
    "superconductor": lambda: SuperconductorDataset(),
    "tf-bind-8": lambda: TFBind8Dataset(),
    "tf-bind-10": lambda: TFBind10Dataset(),
    "dkitty": lambda: DKittyMorphologyDataset(),
    "ant": lambda: AntMorphologyDataset(),
}


def load_elite_data(task_name: str, task: Task) -> Tuple[np.ndarray, np.ndarray]:
    task2naive_name = {
        "Superconductor-RandomForest-v0": "superconductor",
        "AntMorphology-Exact-v0": "ant",
        "DKittyMorphology-Exact-v0": "dkitty",
        "TFBind8-Exact-v0": "tf-bind-8",
        "TFBind10-Exact-v0": "tf-bind-10",
    }
    full_dataset = DATASETS[task2naive_name[task_name]]()
    full_x: np.ndarray = deepcopy(full_dataset.x)
    full_y: np.ndarray = deepcopy(full_dataset.y)

    def create_mask(full_x: np.ndarray, task_x: np.ndarray) -> np.ndarray:
        dtype = [("", full_x.dtype)] * full_x.shape[1]
        full_x_struct = full_x.view(dtype)
        task_x_struct = task_x.view(dtype)

        mask = np.in1d(full_x_struct, task_x_struct)
        return mask

    if task_name == "TFBind10-Exact-v0":
        index = np.random.choice(full_y.shape[0], 30000, replace=False)
        full_x = full_x[index]
        full_y = full_y[index]

    task_x: np.ndarray = task.x
    task_y: np.ndarray = task.y

    mask = create_mask(full_x, task_x)
    diff_x = full_x[~mask]
    diff_x, unique_indices = np.unique(diff_x, axis=0, return_index=True)

    diff_y = full_y[~mask][unique_indices]

    indices = np.arange(diff_x.shape[0])
    np.random.shuffle(indices)
    diff_x = diff_x[indices]
    diff_y = diff_y[indices]

    return diff_x, diff_y


def task_kwargs(task_name: str) -> dict:
    special_task = {
        "TFBind10-Exact-v0": {"dataset_kwargs": {"max_samples": 10000}},
    }
    return special_task.get(task_name, {})


def record_results(
    metric: str,
    task: str,
    model: str,
    seed: int,
    performance: float,
    csv_path: Optional[os.PathLike] = None,
) -> None:
    if csv_path is None:
        result_dir = os.path.join(_BASE_PATH, "results", f"{task}")
        os.makedirs(result_dir, exist_ok=True)
        print(result_dir)

        csv_path = os.path.join(result_dir, f"{metric}-{task}.csv")

    result = {"Model": model, f"{seed}": performance}

    if not os.path.exists(csv_path):
        new_df = pd.DataFrame([result])
        new_df.set_index("Model", inplace=True)
        new_df.columns = new_df.columns.astype(int)
        new_df = new_df.sort_index(axis=1)
        new_df.to_csv(csv_path, index=True)
    else:
        existing_df = pd.read_csv(csv_path, header=0, index_col=0)
        updated_df = existing_df.copy()
        updated_df.loc[f"{model}", f"{seed}"] = performance
        updated_df.columns = updated_df.columns.astype(int)
        updated_df = updated_df.sort_index(axis=1)
        updated_df.to_csv(csv_path, index=True, mode="w")


def record_from_dict(
    metric_dict: Dict[str, float],
    task: str,
    model: str,
    seed: int,
) -> None:
    for metric, performance in metric_dict.items():
        record_results(
            metric=metric,
            task=task,
            model=model,
            seed=seed,
            performance=performance,
        )
