import time
from pathlib import Path
from typing import Optional

import attr
import torch
from matplotlib import pyplot as plt
import pandas as pd

__all__ = ['Logger', ]


@attr.s
class Logger():
    root = attr.ib(type=Path | str)
    info = attr.ib(type=Optional[str], default=None)

    def __attrs_post_init__(self):
        self.info = self.info or time.strftime("%y%m%d%H%M")
        self.path = Path(self.root, self.info)
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_file(self) -> Path:
        return self.path / 'train.pth'

    def draw_loss(self) -> None:
        try:
            epoch_file = self.path / "epoch.csv"
            df = pd.read_csv(epoch_file)
            plt.clf()
            plt.plot(df['epoch'], df['train_loss'], label='train_loss')
            plt.plot(df['epoch'], df['valid_loss'], label='valid_loss')
            plt.legend()
            plt.savefig(self.path / "loss.pdf")
        except:
            pass

    def load_model(self) -> dict | None:
        if self.info == "test":
            return
        if self.checkpoint_file.exists():
            return torch.load(self.checkpoint_file)

    def save_model(self, epoch, model, optimizer, scheduler) -> None:
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, self.checkpoint_file)

    def write_epoch(self, *args):
        epoch_file = self.path / "epoch.csv"
        if not epoch_file.exists():
            epoch_file.write_text("epoch,train_loss,valid_loss,run_time\n")
        assert len(args) == 4
        with epoch_file.open("a") as f:
            f.write(",".join("{}".format(arg) for arg in args))
            f.write("\n")
        self.draw_loss()
