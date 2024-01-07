import shutil
import os
import sys
import logging


libraries = (
    'opencv-python',
    'torch',
    'pandas',
    'torchmetrics',
    'toolz',
    'iteration-utilities',
)

os.system(f'pip install {" ".join(libraries)}')

sys.path.append('../input/img-seg-comp')

import dataset
import analysis


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
os.system('unzip data.zip')

import torch

from torch.utils.tensorboard import SummaryWriter
import utils
from parameters import Parameters, TrainData
from trainer import Trainer
from unet import UNet
from constants import Paths

run_analysis = True

if run_analysis:
    os.system('python analysis.py')

writer = SummaryWriter(TrainData.log_dir)

models = (
    UNet(Parameters.permutated_image_size[0], Parameters.n_classes, depth=3),
)

labels = analysis.load_labels()
normalize = utils.get_normalize(labels)

for model in models:
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainData.lr, weight_decay=TrainData.weight_decay)
    train_data = TrainData(optimizer)
    model.to(Parameters.device)
    trainer = Trainer(model, writer, verbose=3, metrics=train_data.metrics, optimizer=train_data.optimizer, loss=train_data.loss, device=Parameters.device)
    print('Data Path:', Paths.DATA.resolve())
    print('pwd:', os.system('pwd'))
    train_loader, val_loader, test_loader = dataset.get_dataloaders(normalize)
    trainer.train(train_loader, val_loader, test_loader)
