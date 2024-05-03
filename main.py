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

bridge = Bridge()
loader = DataLoader(bridge, 4, True, worker_init_fn=2)
device = torch.device('cuda')
output = Path('~/data/result').expanduser()
logger = Logger(output, 'test')
fit(model, scheduler, optimizer, criterion, loader, loader, loader, 1000, device, output, logger)
