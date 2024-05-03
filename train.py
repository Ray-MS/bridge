from pathlib import Path

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from monai.networks.nets import BasicUNet
from torch.utils.data import DataLoader

from datasets import Bridge
from run import fit, Logger

model = BasicUNet(2, 3, 3)
optimizer = torch.optim.RMSprop(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
criterion = torch.nn.CrossEntropyLoss()

train_set = Bridge(train=True)
train_loader = DataLoader(train_set, 64, True, worker_init_fn=2)
output = Path('~/data/result').expanduser()
logger = Logger(output, 'bridge/unet')

fit(
    logger = logger,
    model = model,
    optimizer = optimizer,
    scheduler = scheduler,
    criterion = criterion,
    train_loader = train_loader,
    valid_loader = train_loader,
    test_loader = train_loader,
    epochs = 100,
    device = torch.device('cuda'),
)
