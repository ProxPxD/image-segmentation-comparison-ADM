import os

from trainer import Trainer
from unet import UNet
import torch
import utils

from torch.utils.tensorboard import SummaryWriter
from parameters import Parameters, TrainData


libraries = (
    'opencv-python',
    'torch',
    'pandas',
    'torchmetrics',
    'pydash',
    'iteration-utilities',
)

os.system(f'pip install {" ".join(libraries)}')
os.system('unzip data.zip')


run_analysis = True

if run_analysis:
    os.system('python analysis.py')

writer = SummaryWriter(TrainData.log_dir)

models = (
    UNet(Parameters.permutated_image_size[0], Parameters.n_classes),
)

for model in models:
    TrainData.optimizer = torch.optim.Adam(model.parameters(), lr=Parameters.lr, weight_decay=Parameters.weight_decay)
    model.to(Parameters.device)
    trainer = Trainer(model, writer, )
    train_loader, val_loader, test_loader = utils.get_dataloaders(utils.normalize)
    trainer.train(train_loader, val_loader, test_loader)
