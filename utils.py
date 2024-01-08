import re
import torchvision
import torch
from itertools import product
from pathlib import Path
from typing import Callable, Iterable

from constants import L
from parameters import Parameters

import cv2 as cv
import numpy as np
from iteration_utilities import starfilter
from toolz import compose_left

import operator as op

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


def path_to_numpy(iterable: Iterable, normalize: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda *args: args) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for img_path, mask_path in iterable:
        load = compose_left(str, cv.imread)
        img, mask = load(img_path), load(mask_path)
        img, mask = normalize(img, mask)
        yield img, mask
  # for img_path, mask_path in iterable:
  #   yield cv.imread(str(img_path)), cv.imread(str(mask_path))


def get_resize(shape):
    torch_resize = torchvision.transforms.Resize(shape)
    resize = compose_left(
        torch.from_numpy,
        torch_resize,
        torch.Tensor.numpy
    )
    return resize


def normalize_mask(mask, label_dict, resize=lambda img: img):
    flat_mask = mask.reshape(-1, mask.shape[-1])
    label_indices = np.array([label_dict[tuple(pixel)] for pixel in flat_mask], dtype=np.int32)
    mask = label_indices.reshape(tuple(mask.shape[:-1]) + (1,))
    print('mask pre', mask.size)
    mask = resize(mask)
    print('mask post', mask.size)
    mask = mask.transpose(1, 2, 0)
    print('mask post transposing', mask.size)
    return mask


def normalize_picture(img: np.ndarray, resize=lambda img: img):
    img = img.astype(np.float32) / 255
    print('img pre', img.size)
    img = img.transpose((2, 0, 1))
    img = resize(img)
    img = img.transpose((1, 2, 0))
    print('post pre', img.size)
    return img


def get_normalize(labels):
    resize = get_resize(Parameters.normalized_image_size[1:])
    label_dict = {tuple(row[L.COLOR]): idx for idx, row in labels.iterrows()}
    return lambda X, mask: (normalize_picture(X, resize=resize), normalize_mask(mask, label_dict, resize=resize))
