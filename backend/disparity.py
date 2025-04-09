import numpy as np
from matplotlib import pyplot as plt

def rgb2gray(img):
    return np.einsum("ijk->ij", img) / 3

def SSD(x, y):
    return np.sum((x - y) ** 2, axis=(1, 2))

# Inputs must be grayscale (and same size)
# Window must be odd dimensions
def disparity_calc(img1, img2, windowDim): 
    HEIGHT = img1.shape[0]
    WIDTH = img1.shape[1]

    depth = np.zeros((img1.shape[0], img1.shape[1], 3))

    # Pad both images to accomodate window
    pad = windowDim // 2
    pad = ((pad, pad), (pad, pad))
    img1 = np.pad(img1, pad, "symmetric")
    img2 = np.pad(img2, pad, "symmetric")

    window1 = np.lib.stride_tricks.sliding_window_view(img1, (3, 3))
    window2 = np.lib.stride_tricks.sliding_window_view(img2, (3, 3))
    for row in range(HEIGHT):
        for col in range(WIDTH): # May be able to get rid of this for loop
            depth[row, col, :] = np.abs(np.argmin(SSD(window1[row, col], window2[row])) - col)

    print(depth[0,0,:])
    plt.imsave('./results/test.jpg', depth / depth.max())


img1 = plt.imread("data/view1.png")
img2 = plt.imread("data/view5.png")
disparity_calc(rgb2gray(img1), rgb2gray(img2), 11)