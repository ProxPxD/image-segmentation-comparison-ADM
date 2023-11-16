from pathlib import Path
import re
from itertools import product, starmap
from typing import Callable, Iterable

from iteration_utilities import starfilter
import numpy as np
import cv2 as cv
from pydash.functions import flow, partial

from constants import Paths


path_like = Path | str


def read_img_mask_name_pairs(
        img_path: path_like,
        mask_path: path_like = None,
        img_pattern=None,
        mask_pattern=None,
        is_sorted_pairwise=True,
        img_and_mask_match_cond: Callable[[path_like, path_like], bool] = None,
) -> Iterable[tuple[Path, Path]]:
    mask_path = mask_path or img_path
    img_path = Path(mask_path) if mask_path else img_path
    compiled_mask_pattern = re.compile(mask_pattern or r'.png$')
    mask_cond = lambda mask: compiled_mask_pattern.search(str(mask))
    if img_path == mask_path and img_pattern is None and mask_pattern is not None:
        img_cond = lambda img: not mask_cond(img)
    else:
        compiled_img_pattern = re.compile(img_pattern or r'.png$')
        img_cond = lambda img: compiled_img_pattern.search(str(img))
    imgs = filter(img_cond, img_path.iterdir())
    masks = filter(mask_cond, mask_path.iterdir())
    if is_sorted_pairwise:
        # return zip(img_path.iterdir(), mask_path.iterdir())
        return zip(sorted(imgs), sorted(masks))

    if img_and_mask_match_cond is None:
        img_and_mask_match_cond = lambda img, mask: img.stem in mask.stem
    return starfilter(img_and_mask_match_cond, product(imgs, masks))


def path_to_numpy(iterable: Iterable, normalize: Callable[[np.ndarray], np.ndarray] = lambda *args: args) -> Iterable[tuple[np.ndarray, np.ndarray]]:
  return starmap(normalize, map(flow(partial(map, flow(str, cv.imread)), tuple), iterable))
  # for img_path, mask_path in iterable:
  #   yield cv.imread(str(img_path)), cv.imread(str(mask_path))



def get_dataloaders():
    path_tuples = list(utils.read_img_mask_name_pairs(Paths.INPUT_IMGAGES, mask_pattern=r'_L.png$', is_sorted_pairwise=True))
    dataset_paths = torch.utils.data.random_split(path_tuples, Parameters.dataset_persentages)
    datasets = map(CamSeqDS, dataset_paths)
    loaders = tuple(map(lambda ds: DataLoader(ds, batch_size=Parameters.batch_size), datasets))
    return loaders

