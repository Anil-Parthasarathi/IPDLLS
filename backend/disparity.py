import numpy as np
from matplotlib import pyplot as plt
import time

def rgb2grayInt(img):
    rgb = np.sum(img, axis=2) / 3
    return rgb / rgb.max()

def SSD1(x, y):
    return np.sum((x - y) ** 2, axis=(1, 2))

def SSD2(x, y):
    return np.sum((x - y) ** 2, axis=(2, 3))

def SSD3(x, y):
    return np.sum((x - y) ** 2, axis=(3, 4))

# Inputs must be grayscale (and same size)
# Window must be odd dimensions
def two(img1, img2, windowDim): 
    HEIGHT = img1.shape[0]
    WIDTH = img1.shape[1]

    depth = np.zeros_like(img1)

    # Pad both images to accomodate window
    pad = windowDim // 2
    pad = ((pad, pad), (pad, pad))
    img1 = np.pad(img1, pad, "symmetric")
    img2 = np.pad(img2, pad, "symmetric")

    window1 = np.lib.stride_tricks.sliding_window_view(img1, (windowDim, windowDim))
    window2 = np.lib.stride_tricks.sliding_window_view(img2, (windowDim, windowDim))
    for row in range(HEIGHT):
        for col in range(WIDTH):
            depth[row, col] = np.argmin(SSD1(window1[row, col], window2[row])) - col

    depth = abs(depth)
    depth = np.stack((depth, depth, depth), axis=-1)
    plt.imsave('./results/0_two.jpg', depth / depth.max())

def one(img1, img2, windowDim): 
    HEIGHT = img1.shape[0]
    WIDTH = img1.shape[1]

    depth = np.ones_like(img1)
    depth[:, 0] = 0
    depth = np.cumsum(depth, axis=1, dtype=img1.dtype)

    # Pad both images to accomodate window
    pad = windowDim // 2
    pad = ((pad, pad), (pad, pad))
    img1 = np.pad(img1, pad, "symmetric")
    img2 = np.pad(img2, pad, "symmetric")
    
    # Calculate Disparities
    window1 = np.lib.stride_tricks.sliding_window_view(img1, (windowDim, windowDim))
    window2 = np.lib.stride_tricks.sliding_window_view(img2, (windowDim, windowDim))
    for row in range(HEIGHT):
        depth[row] -= np.argmin(SSD2(window1[row][:, np.newaxis], window2[row]), axis=1)

    depth = abs(depth)
    depth = np.stack((depth, depth, depth), axis=-1)
    plt.imsave('./results/0_one.jpg', depth / depth.max())

def zero(img1, img2, windowDim): 
    HEIGHT = img1.shape[0]
    WIDTH = img1.shape[1]

    depth = np.ones_like(img1)
    depth[:, 0] = 0
    depth = np.cumsum(depth, axis=1, dtype=img1.dtype)

    # Pad both images to accomodate window
    pad = windowDim // 2
    pad = ((pad, pad), (pad, pad))
    img1 = np.pad(img1, pad, "symmetric")
    img2 = np.pad(img2, pad, "symmetric")

    # Calculate Disparities
    window1 = np.lib.stride_tricks.sliding_window_view(img1, (windowDim, windowDim))
    window2 = np.lib.stride_tricks.sliding_window_view(img2, (windowDim, windowDim))

    depth -= np.argmin(SSD3(window1[:][:, :, np.newaxis], window2[:][:, np.newaxis]), axis=2)

    depth = abs(depth)
    depth = np.stack((depth, depth, depth), axis=-1)
    plt.imsave('./results/0_zero.jpg', depth / depth.max())

# A = np.array([
#     [0,0,0,0],
#     [0,4,0,0],
#     [0,0,0,0]
# ])
# B = np.array([
#     [0,0,0,0],
#     [0,1,4,0],
#     [0,0,0,0]   
# ])
# base(A, B, 3)
# print('--------------------')
# working(A, B, 3)
# print('--------------------')
# zero(A, B, 3)

rows = 500
cols = 500
w = 3
img1 = rgb2grayInt(plt.imread("data/view1.png"))[: rows, : cols]
img2 = rgb2grayInt(plt.imread("data/view5.png"))[: rows, : cols]

print("Img shape:", img1.shape, "Window:", w)
print("Two for loops time: ", end='', flush=True)
start = time.time()
two(img1, img2, w)
print(time.time() - start)

print("One for loops time: ", end='', flush=True)
start = time.time()
one(img1, img2, w)
print(time.time() - start)

print("Zero for loops time: ", end='', flush=True)
start = time.time()
zero(img1, img2, w)
print(time.time() - start)
