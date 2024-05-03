import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

from .logger import Logger


__all__ = ["train", "valid", "test", "fit", ]

def imsave(arr, path):
    import skimage.io
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.uint8:
            skimage.io.imsave(path, arr)
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr = np.array(arr * 255, dtype=np.uint8)
        skimage.io.imsave(path, arr)


def train(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device,
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
    loader,
    criterion,
    device,
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
    model: nn.Module,
    loader: DataLoader,
    device,
    output: Path,
    epoch: int,
):
    model.eval()
    folder = output / f'{epoch}'
    folder.mkdir(exist_ok=True)
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        for i, (data, gt, pred) in enumerate(zip(x, y, out)):
            index = idx * loader.batch_size + i

            data = T.ToPILImage()(data)
            data.save(folder / f'{index:03d}_x.png')

            gt = gt.cpu().numpy()[0].astype(np.uint8)
            Image.fromarray(gt * 126, mode='L').save(folder / f'{index:03d}_y.png')

            pred = pred.cpu().numpy().argmax(0).astype(np.uint8)
            Image.fromarray(pred * 126, mode='L').save(folder / f'{index:03d}_z.png')


def fit(
        model, scheduler, optimizer, criterion,
        train_loader, valid_loader, test_loader,
        epochs, device, output_root, logger: Logger
):
    model = model.to(device)
    cp = logger.load_model()
    e0 = 0
    if cp is not None:
        e0 = cp[0]
        model.load_state_dict(cp[1])
        scheduler.load_state_dict(cp[2])
        optimizer.load_state_dict(cp[3])

    for epoch in range(e0, epochs):
        t0 = time.time()
        print("epoch=>{}:".format(epoch + 1))
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print("\ttrain=>loss:{}".format(train_loss))
        valid_loss = valid(model, valid_loader, criterion, device)
        print("\tvalid=>loss:{}".format(valid_loss))
        scheduler.step()
        t1 = time.time()
        logger.write_epoch(epoch + 1, train_loss, valid_loss, t1 - t0)
        logger.save_model(epoch + 1, model, scheduler, optimizer)

        test(model, test_loader, device, logger.path, epoch)
