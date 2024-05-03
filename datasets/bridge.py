from pathlib import Path

import monai.transforms as MT
import numpy as np
import torch
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ['Bridge', ]


class Bridge(Dataset):
    def __init__(
        self,
        root: Path | str = '~/data/data',
        train: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root, 'bridge').expanduser()
        self.train = train

        if train:
            self.data, self.targets = self._load_train_data()
            return
        self.data = self._load_valid_data()

    def _load_train_data(self) -> tuple[list, list]:
        data, targets = [], []
        for file in self.label_folder.iterdir():
            if not file.name.endswith('.png'): continue

            label = Image.open(file).convert('L')
            label = np.array(label)
            label_ = np.zeros(label.shape, np.uint8)
            label_[label == 255] = 1
            label_[label == 127] = 2
            label = torch.from_numpy(label_)
            targets.append(label)

            image = Image.open(self.image_folder / file.name)
            image = rearrange(np.array(image), 'h w c -> c h w')
            image = torch.from_numpy(image).div(255)
            data.append(image)
        return data, targets

    @property
    def image_folder(self) -> Path:
        return self.root / 'video'

    @property
    def label_folder(self) -> Path:
        return self.root / 'label'

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img, seg = self.data[index], self.targets[index]
        seg = seg.unsqueeze(0)
        data_dict = {'img': img, 'seg': seg, }
        transform = MT.Compose([
            MT.RandSpatialCropD(data_dict.keys(), 1280),
            MT.ResizeD(data_dict.keys(), (224, 224))
        ])
        data_dict = transform(data_dict)
        img = data_dict['img']
        seg = data_dict['seg'].to(torch.int64)
        return img, seg

    def __len__(self) -> int:
        return len(self.data)
