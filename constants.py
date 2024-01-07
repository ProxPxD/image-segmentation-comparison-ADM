from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    DATA = Path('../input/img-seg-comp/data/data')
    INPUT_DATA = DATA / 'input'
    INPUT_IMGAGES = INPUT_DATA / 'img'
    INPUT_LABELS = INPUT_DATA / 'label_colors.txt'

    OUTPUT_DATA = DATA / 'output'


@dataclass
class Labels:
  COLOR = 'Color'
  CLASS_NAME = 'Class-Name'

L = Labels
