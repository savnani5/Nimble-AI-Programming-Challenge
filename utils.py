import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from collections import OrderedDict
from PIL import Image


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def rgb2array(data_path,
              desired_size=None, 
              expand=False,
              hwc=True,
              show=False):
    """Loads a 24-bit PNG RGB image as a 3D or 4D numpy array."""
    img = Image.open(data_path).convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    x = np.array(img, dtype=np.float32)
    if show:
        plt.imshow(x.astype(np.uint8), interpolation='nearest')
    if not hwc:
        x = np.transpose(x, [2, 0, 1])
    if expand:
        x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    return x


def depth2array(data_path,
                desired_size=None,
                expand=False,
                hwc=True,
                depth_scale=1e-3,
                show=False):
    """Loads a 16-bit PNG depth image as a 3D or 4D numpy array."""
    img = Image.open(data_path)
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    x = np.array(img, dtype=np.float32)
    if show:
        plt.imshow(x, cmap='gray', interpolation='nearest')
    x = np.expand_dims(x, axis=0)
    if hwc:
        x = np.transpose(x, [1, 2, 0])
    if expand:
        x = np.expand_dims(x, axis=0)
    # Intel RealSense depth cameras store
    # depth in millimers (1e-3 m) so we convert
    # back to meters
    x *= depth_scale
    x = x.astype(np.float32)
    return x


def label2array(data_path, desired_size=None, hwc=True, show=False):
    """Loads an 8-bit grayscale PNG image as a 3D numpy array."""
    img = Image.open(data_path)
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    x = np.array(img, dtype=np.int64)[..., 1]
    if show:
        plt.imshow(x, norm=MidpointNormalize(0, 255, 1), interpolation='nearest')
        # plt.imshow(x, cmap=plt.get_cmap('hot'), interpolation='nearest', vmin=0, vmax=255)
    x = np.expand_dims(x, axis=0)
    if hwc:
        x = np.transpose(x, [1, 2, 0])
    x[x == 255] = 1.
    return x


if __name__=="__main__":
    c = rgb2array("data/train/color/938_0.png")
    c = c/np.linalg.norm(c)
    d = depth2array("data/train/depth/938_0.png")
    l = label2array("data/train/label/938_0.png")
    # x = np.multiply(d,l)

    print(l.shape)
    
    # print("color", np.unique(c))
    # print("depth", np.unique(d))
    # print("label", np.unique(l))
    # print("mul", np.unique(x))  
    # dest = cv2.addWeighted(c, 0.5, x, 0.5, 0.0)
    # plt.imshow(dest, cmap='plasma', interpolation='nearest') 
    # plt.show()