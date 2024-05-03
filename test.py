from datasets import Bridge
import torchvision.transforms as T
from PIL import Image

bridge = Bridge()
for idx, (x, y) in enumerate(bridge):
    x = T.ToPILImage()(x)
    x.save(f'{idx}.png')
