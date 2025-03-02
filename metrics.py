import numpy as np
import torch

_TKWARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


@torch.no_grad()
def spearman_corr(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:

    y1 = y1.reshape(
        -1,
    )
    y2 = y2.reshape(
        -1,
    )
    assert len(y1) == len(y2)

    y1_rank = y1.argsort().argsort()
    y2_rank = y2.argsort().argsort()

    mean_y1_rank = torch.mean(y1_rank.float())
    mean_y2_rank = torch.mean(y2_rank.float())

    cov = torch.mean((y1_rank - mean_y1_rank) * (y2_rank - mean_y2_rank))
    std_y1 = torch.sqrt(torch.mean((y1_rank - mean_y1_rank) ** 2))
    std_y2 = torch.sqrt(torch.mean((y2_rank - mean_y2_rank) ** 2))

    corr = cov / (std_y1 * std_y2)
    return corr


@torch.no_grad()
def calculate_percentage_overlap_torch(indices1, indices2) -> torch.Tensor:
    n = indices1.size(0)
    percentages = torch.zeros(n).to(**_TKWARGS)

    for i in range(1, n + 1):
        sub_indices1 = indices1[:i].unsqueeze(1)  # shape [i, 1]
        sub_indices2 = indices2[:i].unsqueeze(0)  # shape [1, i]

        intersection_matrix = sub_indices1 == sub_indices2  # shape [i, i]
        intersection_count = intersection_matrix.any(dim=0).sum().float()

        percentages[i - 1] = intersection_count / i

    return percentages


@torch.no_grad()
def trapz_torch(y, dx=1.0) -> np.ndarray:
    n = y.size(0)
    y_avg = (y[:-1] + y[1:]) / 2
    area = torch.sum(y_avg) * dx
    return area.detach().cpu().numpy()


@torch.no_grad()
def cal_overlap_auc(
    y1: torch.Tensor, y2: torch.Tensor, max_samples: int = 30000
) -> np.ndarray:
    y1 = y1.reshape(
        -1,
    )
    y2 = y2.reshape(
        -1,
    )
    assert len(y1) == len(y2), "Input tensors must have the same length"

    if len(y1) > max_samples:
        indices = torch.randperm(len(y1))[:max_samples]
        y1 = y1[indices]
        y2 = y2[indices]

    indices_1 = torch.argsort(y1)
    indices_2 = torch.argsort(y2)
    data = calculate_percentage_overlap_torch(indices_1, indices_2)

    area_under_curve = trapz_torch(data, dx=1 / len(y1))

    return area_under_curve
