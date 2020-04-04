#!/usr/bin/env python3
# coding: utf-8

import random
from PIL import Image

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img, logger):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            logger.debug("flipped the image")
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        logger.debug("did not filp the image")
        return img

    def randomize_parameters(self):
        self.p = random.random()


if __name__ == "__main__":
    pass
