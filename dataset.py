import numpy as np
from torch.utils.data import IterableDataset, DataLoader

import torch
from constants import Paths
from parameters import Parameters
import utils


class CamSeqDS(IterableDataset):
    def __init__(self, path_tuples, normalize=lambda args: args):
        self.path_tuples = path_tuples
        self.normalize = normalize
        self.image_transposition = (2, 0, 1)

    def __iter__(self):
        numpied = utils.path_to_numpy(self.path_tuples, self.normalize)
        for img, mask in numpied:
            print(self.image_transposition)
            print('img:', img)
            print('mask:', mask)
            print('img.T:', img.transpose(self.image_transposition))
            print('mask.T:', mask.transpose(self.image_transposition))
            yield img.transpose(self.image_transposition), mask
        return
        transposed = map(lambda arr: tuple((a.transpose(self.image_transposition) for a in arr)), numpied)
        yield from transposed


def get_dataloaders(normalize=lambda args: args):
    path_tuples = list(utils.read_img_mask_name_pairs(Paths.INPUT_IMGAGES, mask_pattern=r'_L.png$', is_sorted_pairwise=True))
    dataset_paths = torch.utils.data.random_split(path_tuples, Parameters.dataset_persentages)
    datasets = map(lambda paths: CamSeqDS(paths, normalize), dataset_paths)
    loaders = tuple(map(lambda ds: DataLoader(ds, batch_size=Parameters.batch_size), datasets))
    return loaders
