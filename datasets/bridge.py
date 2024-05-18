import random
from pathlib import Path

import cv2
import monai.transforms as MT
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from torch import Tensor
from torchvision.datasets import VisionDataset

from ._factory import register_dataset

__all__ = [
    'Bridge',
    'BridgePier',
    'BridgeRuler',
    'BridgeTest',
    'BridgeTest_v2',
]


class Bridge(VisionDataset):
    def __init__(
        self,
        root: Path | str = '~/data/data',
        train: bool = True,
        reload: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root, 'bridge').expanduser()
        self.train = train

        if self._check_legacy_data() and not reload:
            self.data, self.targets = self._load_legacy_data()
            return
        self.data, self.targets = self._load_data()

    def _check_legacy_data(self) -> bool:
        return self.train_file.exists()

    def _load_legacy_data(self) -> tuple[list, list]:
        data = torch.load(self.train_file)
        return data['img'], data['seg']

    def _load_data(self) -> tuple[list, list]:
        self.processed_folder.mkdir(exist_ok=True)

        folder = self.root / 'output'
        data, targets = [],  []
        for file in Path(folder, 'JPEGImages').glob('*.jpg'):
            img = Image.open(file)
            data.append(img)
            target_file = folder / 'SegmentationClassNpy' / f'{file.stem}.npy'
            target = np.load(target_file)
            targets.append(target)
        torch.save({'img': data, 'seg': targets}, self.train_file)
        return data, targets

    @property
    def processed_folder(self) -> Path:
        return self.root / 'processed'

    @property
    def train_file(self) -> Path:
        return self.processed_folder / 'train.pt'

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img, target = self.data[index], self.targets[index]
        img = TF.to_tensor(img)
        seg = torch.from_numpy(target).unsqueeze(0)

        crop = T.RandomCrop(1280)
        i, j, h, w = crop.get_params(img, (1280, 1280))
        img = TF.crop(img, i, j, h, w)
        seg = TF.crop(seg, i, j, h, w)

        img = TF.resize(img, 224)
        seg = TF.resize(seg, 224, TF.InterpolationMode.NEAREST)
        
        return img, seg.to(torch.int64)

    def __len__(self) -> int:
        return len(self.data)


class BridgePier(Bridge):
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img, seg = super().__getitem__(index)
        seg[seg == 3] = 1
        return img, seg


class BridgeRuler(Bridge):
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img, seg = super().__getitem__(index)
        seg = torch.where(seg == 3, 1, 0)
        return img, seg


class BridgeTest(VisionDataset):
    def __init__(
        self,
        root: Path | str = '~/data/data',
        video: str = '1713526207',
        len: int = 1000,
    ) -> None:
        self.root = Path(root, 'bridge').expanduser()
        self.video = self.root / f'{video}.mp4'
        self.cap = cv2.VideoCapture(self.video.as_posix())
        self.idx = list(random.randint(0, 40000) for _ in range(len))

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        frame = self.idx[index]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = self.cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = T.ToTensor()(img)
        transform = MT.Compose([
            MT.CenterSpatialCrop(1280),
            MT.Resize((224, 224)),
        ])
        return transform(img), frame

    def __len__(self) -> int:
        return len(self.idx)


class BridgeTest_v2(VisionDataset):
    def __init__(
        self,
        root: Path | str = '~/data/data',
    ) -> None:
        self.root = Path(root, 'bridge').expanduser()
        self.data = self._load_data()

    def _load_data(self) -> list:
        folder = self.root / '0504'
        data = []
        for file in folder.iterdir():
            if not file.name.endswith('.jpg'):
                continue
            image = Image.open(file)
            image = rearrange(np.array(image), 'h w c -> c h w')
            image = torch.from_numpy(image).div(255)
            data.append(image)
        return data

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        transform = MT.Compose([
            MT.CenterSpatialCrop(1280),
            MT.Resize((224, 224)),
        ])
        return transform(self.data[index]), index

    def __len__(self) -> int:
        return len(self.data)
