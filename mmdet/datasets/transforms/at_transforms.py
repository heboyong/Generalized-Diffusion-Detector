import random
from typing import Dict

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFilter
from mmcv.transforms.base import BaseTransform

from mmdet.registry import TRANSFORMS


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


@TRANSFORMS.register_module()
class at_strong_augmentation(BaseTransform):

    def build_strong_augmentation(self):
        """
        Create a list of :class:`Augmentation` from config.
        Now it includes resizing and flipping.

        Returns:
            list[Augmentation]
        """

        augmentation = []

        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                ),
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                ),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)
        return transforms.Compose(augmentation)

    def transform(self, results: Dict) -> Dict:
        """Call function to add random Gaussian blur to image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with random Gaussian noise.
        """
        img = results['img']

        seq = self.build_strong_augmentation()
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_image = seq(pil_image)
        img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class at_weak_augmentation(BaseTransform):

    def build_weak_augmentation(self):
        """
        Create a list of :class:`Augmentation` from config.
        Now it includes resizing and flipping.

        Returns:
            list[Augmentation]
        """

        augmentation = []

        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.1))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.3))

        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.5, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                ),
                transforms.RandomErasing(
                    p=0.1, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                ),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)
        return transforms.Compose(augmentation)

    def transform(self, results: Dict) -> Dict:
        """Call function to add random Gaussian blur to image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with random Gaussian noise.
        """
        img = results['img']

        seq = self.build_weak_augmentation()
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_image = seq(pil_image)
        img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
