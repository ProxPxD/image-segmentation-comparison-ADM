import shutil
import os
import sys
import logging

sys.path.append('../input/img-seg-comp')

import dataset


# def create_logger(logger_name):
#     # create logger
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging.DEBUG)
# 
#     formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')
# 
#     # create handler
#     streamHandler = logging.StreamHandler()
#     streamHandler.setLevel(logging.DEBUG)
#     streamHandler.setFormatter(formatter)
# 
#     logger.addHandler(streamHandler)
#     return logger
# 
# logger = create_logger('dupa')
# 
# logger.debug(f'modules: {sys.modules.keys()}')
# logger.info(f'modules: {sys.modules.keys()}')


libraries = (
    'opencv-python',
    'torch',
    'pandas',
    'torchmetrics',
    'toolz',
    'iteration-utilities'
)

os.system(f'pip install {" ".join(libraries)}')
os.system('unzip data.zip')

import torch

from torch.utils.tensorboard import SummaryWriter
import utils
from parameters import Parameters, TrainData
from trainer import Trainer
from unet import UNet

run_analysis = True

if run_analysis:
    os.system('python analysis.py')

writer = SummaryWriter(TrainData.log_dir)

models = (
    UNet(Parameters.permutated_image_size[0], Parameters.n_classes),
)

for model in models:
    TrainData.optimizer = torch.optim.Adam(model.parameters(), lr=TrainData.lr, weight_decay=TrainData.weight_decay)
    model.to(Parameters.device)
    trainer = Trainer(model, writer, )
    train_loader, val_loader, test_loader = dataset.get_dataloaders(utils.normalize)
    trainer.train(train_loader, val_loader, test_loader)
