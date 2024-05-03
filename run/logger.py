import time
from pathlib import Path
from typing import Optional

import attr
import torch
from matplotlib import pyplot as plt
import pandas as pd


@attr.s
class Logger:
    root = attr.ib(type=Path | str)
    info = attr.ib(type=Optional[str], default=None)

    def __attrs_post_init__(self):
        self.info = self.info or time.strftime("%y%m%d%H%M")
        self.path = Path(self.root, self.info)
        self.path.mkdir(parents=True, exist_ok=True)

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

    def load_model(self) -> Optional[tuple[int, dict, dict, dict]]:
        if self.info == "test":
            return
        file = self.path / "train.pth"
        if file.exists():
            return torch.load(file)

    def save_model(self, epoch, model, scheduler, optimizer) -> None:
        file = self.path / "train.pth"
        torch.save((epoch, model.state_dict(), scheduler.state_dict(), optimizer.state_dict()), file)

    def write_epoch(self, *args):
        epoch_file = self.path / "epoch.csv"
        if not epoch_file.exists():
            epoch_file.write_text("epoch,train_loss,valid_loss,run_time\n")
        assert len(args) == 4
        with epoch_file.open("a") as f:
            f.write(",".join("{}".format(arg) for arg in args))
            f.write("\n")
        self.draw_loss()
