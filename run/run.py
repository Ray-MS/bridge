import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader

from .logger import Logger

__all__ = ['fit', 'train', 'valid', 'test', ]


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device = torch.device('cuda'),
):
    model.train()
    loss_total = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y.squeeze(1))
        loss_total += loss
        loss.backward()
        optimizer.step()
    return loss_total / len(loader)


@torch.no_grad()
def valid(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device('cuda'),
):
    model.eval()
    loss_total = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y.squeeze(1))
        loss_total += loss
    return loss_total / len(loader)


@torch.no_grad()
def test(
    logger: Logger,
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = torch.device('cuda'),
):
    model.eval()
    folder = logger.path / f'{epoch:04d}'
    folder.mkdir(exist_ok=True)
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        out = model(x)

        for idx, (data, frame, pred) in enumerate(zip(x, y, out)):
            index = batch_idx * loader.batch_size + idx

            data = TF.resize(data, 1280)
            data = TF.to_pil_image(data)
            data.save(folder / f'{frame:03d}_x.png')

            pred = TF.resize(pred, 1280)
            pred = pred.cpu().numpy().argmax(0).astype(np.uint8)
            Image.fromarray(
                pred * 126, mode='L').save(folder / f'{frame:03d}_z.png')


def fit(
    logger: Logger,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader | None,
    epochs: int,
    device: torch.device,
):
    model = model.to(device)
    checkpoint = logger.load_model()
    e0 = 0
    if checkpoint is not None:
        e0 = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(e0, epochs):
        t0 = time.time()
        print('epoch=>{}:'.format(epoch + 1))
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print('\ttrain=>loss:{}'.format(train_loss))
        valid_loss = valid(model, valid_loader, criterion, device)
        print('\tvalid=>loss:{}'.format(valid_loss))
        scheduler.step()
        t1 = time.time()
        logger.write_epoch(epoch + 1, train_loss, valid_loss, t1 - t0)
        logger.save_model(epoch + 1, model, optimizer, scheduler)

        if test_loader is None or epoch % 50 > 0:
            continue
        test(logger, epoch, model, test_loader, device)
    test(logger, e0, model, test_loader, device)
