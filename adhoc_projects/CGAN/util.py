import numpy as np
import matplotlib.pyplot as plt

IMG_WIDTH = 28
IMG_HEIGHT = 28


def visualize(img_arr):
    plt.imshow((img_arr.asnumpy().reshape(IMG_WIDTH, IMG_HEIGHT)
                * 255).astype(np.uint8), cmap="gray")
    plt.axis("off")
    return
