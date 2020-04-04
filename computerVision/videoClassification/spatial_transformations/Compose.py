#!/usr/bin/env python3
# coding: utf-8

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, logger):
        self.transforms = transforms
        self.logger = logger
        self.logger.debug("Composing spatial transformations")

    def __call__(self, img):
        for transformation in self.transforms:
            img = transformation(img, self.logger)
        return img

    def randomize_parameters(self):
        for transformation in self.transforms:
            transformation.randomize_parameters()
        self.logger.debug("Randomized the Parameters")


if __name__ == "__main__":
    pass
