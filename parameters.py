from dataclasses import dataclass, asdict, field
from typing import Callable

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.losses import JaccardLoss


@dataclass
class DatasetPercentages:
  train: float = .8
  val: float = .1
  test: float = .1


@dataclass
class Parameters:
    n_classes = 32
    image_original_size = (720, 960, 3)
    permutated_image_size = (3, 720, 960)

    normalized_image_size = (3, 360, 480)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_persentages = tuple(asdict(DatasetPercentages()).values())
    batch_size = 10

    random_split_seed = 42


@dataclass
class TrainData:

    optimizer: torch.optim.Optimizer # = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # loss: Callable = torchmetrics.detection.iou.IntersectionOverUnion(num_classes=Parameters.n_classes)
    # loss: Callable = torchmetrics.JaccardIndex(task='multiclass', num_classes=Parameters.n_classes).to(Parameters.device)
    loss: Callable = lambda outputs, labels: F.cross_entropy(outputs.squeeze(), labels.type(torch.LongTensor).to(Parameters.device))

    lr: float = .005
    weight_decay: float =1e-5
    batch_size: int = 5

    log_dir = 'logs'
    get_model_path: Callable[[str, int, int], str] = lambda model_name, epoch, it: f'{model_name}_{epoch}' + (f'_{it}' if it is not None else '')
    verbose: int = 4

    metrics: dict = field(default_factory=lambda: {
        'Accuracy': torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=Parameters.n_classes),
        'Precision': torchmetrics.Precision(task='multiclass', average='macro', num_classes=Parameters.n_classes),
        'F1': torchmetrics.F1Score(task='multiclass', average='macro', num_classes=Parameters.n_classes),
        'Recall': torchmetrics.Recall(task='multiclass', average='macro', num_classes=Parameters.n_classes),
        'IoU': torchmetrics.JaccardIndex(task='multiclass', average='macro', num_classes=Parameters.n_classes)
        # 'Cross-Entropy': None
    })


if sum(Parameters.dataset_persentages) != 1.0:
    raise ValueError


