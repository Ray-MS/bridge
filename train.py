from pathlib import Path

import torch
from monai.networks.nets import BasicUNet
from torch.utils.data import DataLoader

from datasets import *
from run import fit, Logger

# Pier
model = BasicUNet(2, 3, 3)
optimizer = torch.optim.RMSprop(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
criterion = torch.nn.CrossEntropyLoss()

pier = BridgePier()
loader = DataLoader(pier, 64, True, worker_init_fn=2)
output = Path('~/data/result').expanduser()
logger = Logger(output, 'bridge/pier')

fit(
    logger=logger,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    train_loader=loader,
    valid_loader=loader,
    test_loader=None,
    epochs=3000,
    device=torch.device('cuda'),
)

# Ruler
model = BasicUNet(2, 3, 2)
optimizer = torch.optim.RMSprop(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
criterion = torch.nn.CrossEntropyLoss()

pier = BridgeRuler()
loader = DataLoader(pier, 64, True, worker_init_fn=2)
output = Path('~/data/result').expanduser()
logger = Logger(output, 'bridge/ruler')

fit(
    logger=logger,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    train_loader=loader,
    valid_loader=loader,
    test_loader=None,
    epochs=3000,
    device=torch.device('cuda'),
)
