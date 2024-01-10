import shutil
import os
import sys
from enum import Enum
import logging

import cv2
import numpy as np

libraries = (
    'opencv-python',
    'torch',
    'pandas',
    'torchmetrics',
    'toolz',
    'iteration-utilities',
    'segmentation-models-pytorch'
)

os.system(f'pip install {" ".join(libraries)}')

sys.path.append('../input/img-seg-comp')
os.makedirs('./models')


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
# logger = create_logger('logger')
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


models = {
    'unet': UNet(Parameters.permutated_image_size[0], Parameters.n_classes, depth=3),
}

labels = analysis.load_labels()
normalize = utils.get_normalize(labels)

Mode = Enum('Mode', ['TRAIN', 'TEST'])
mode = Mode.TEST

train_loader, val_loader, test_loader = dataset.get_dataloaders(normalize)

for name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainData.lr, weight_decay=TrainData.weight_decay)
    train_data = TrainData(optimizer)
    model.to(Parameters.device)
    trainer = Trainer(
        model,
        writer=writer,
        model_name=name,
        get_model_path=lambda model_name, epoch, iteration: f'models/{model_name}_e{epoch}',
        verbose=3,
        metrics=train_data.metrics,
        optimizer=train_data.optimizer,
        loss=train_data.loss,
        validate_every_n_epoch=1,
        device=Parameters.device
    )
    trainer.load(4, 4)
    match mode:
        case Mode.TRAIN:
            trainer.train(train_loader, val_loader)
        case Mode.TEST:
            last_index = 0
            os.mkdir('predictions')
            for X, results in test_loader:
                preds = trainer.model(X.to(Parameters.device))
                curr_index = test_loader.dataset.index
                indices = range(last_index, curr_index+1)
                for index, pred in zip(indices, preds):
                    class_mask = np.argmax(pred.cpu().detach().numpy(), axis=0).astype(np.uint8)
                    print('argmax:', class_mask)
                    color_mask = np.fromiter((labels.loc[value] for value in np.nditer(class_mask)), class_mask.dtype).reshape(*class_mask.shape[:-1])
                    print(f'Color mask dimensions: {color_mask.shape}')
                    print('color mask:', color_mask)

                    img_path = test_loader.dataset.path_tuples[index][0]
                    new_mask_name = img_path.replace('.png', '_pred.png').rsplit('/', 1)[-1]
                    new_mask_path = 'predictions/' + new_mask_name
                    print(f'Saving in {new_mask_path}')
                    cv2.imwrite(new_mask_path, color_mask)
        case _:
            print(f'Mode "{mode}" is unknown')
